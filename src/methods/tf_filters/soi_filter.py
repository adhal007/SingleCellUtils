"""
Complete Hierarchical Score based O-Information Pipeline
=============================================

Full implementation with VP-SDE score network and hierarchical O-information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
from scipy.sparse import issparse
from scipy.spatial.distance import jensenshannon
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import time
import src.Models.TFFilters.tf_base as tf_base

# ============================================================================
# VP-SDE COMPONENTS
# ============================================================================

class VPSDE_Fast:
    """GPU-accelerated VP-SDE"""
    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0, device='cuda'):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.device = device
    
    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def alpha(self, t):
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
        return torch.exp(-0.5 * integral)
    
    def sigma(self, t):
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
        sigma_squared = 1 - torch.exp(-integral)
        return torch.sqrt(sigma_squared)
    
    def add_noise(self, x0, t, noise=None):
        t = t.view(-1, 1)
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        
        if noise is None:
            noise = torch.randn_like(x0)
        
        x_t = alpha_t * x0 + sigma_t * noise
        return x_t, noise


class ScoreMatchingDataset(Dataset):
    def __init__(self, data, T=1.0):
        if isinstance(data, np.ndarray):
            self.data = torch.FloatTensor(data)
        else:
            self.data = data
        self.T = T
        self.n_samples = len(self.data)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        x0 = self.data[idx]
        t = torch.rand(1).squeeze() * self.T
        return x0, t


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.embedding_dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.SiLU()
    
    def forward(self, x):
        return self.activation(x + self.block(x))


class FastScoreNetwork(nn.Module):
    def __init__(self, data_dim, hidden_dim=256, time_embed_dim=64, n_layers=4, dropout=0.1):
        super().__init__()
        
        self.data_dim = data_dim
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        
        input_dim = data_dim + time_embed_dim + data_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, data_dim)
        
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x_t, t, mask):
        t_embed = self.time_embed(t)
        h = torch.cat([x_t, t_embed, mask], dim=-1)
        h = self.input_proj(h)
        
        for block in self.blocks:
            h = block(h)
        
        noise_pred = self.output_proj(h)
        return noise_pred


def train_score_network_fast(
    data,
    score_net,
    vpsde,
    n_epochs=100,
    batch_size=512,
    lr=1e-3,
    device='cuda',
    grad_clip=1.0,
    use_amp=True,
    verbose=True
):
    """Fast training with mixed precision"""
    
    score_net = score_net.to(device)
    optimizer = torch.optim.AdamW(score_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )
    
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    dataset = ScoreMatchingDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    if verbose:
        print(f"  Training score network: {n_epochs} epochs, batch {batch_size}")
    
    score_net.train()
    loss_history = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for x0, t in dataloader:
            x0 = x0.to(device)
            t = t.to(device)
            
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                noise = torch.randn_like(x0)
                x_t, _ = vpsde.add_noise(x0, t, noise)
                mask = torch.ones_like(x0)
                noise_pred = score_net(x_t, t, mask)
                loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad(set_to_none=True)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(score_net.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(score_net.parameters(), grad_clip)
                optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        
        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"    Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.6f}")
    
    return score_net, loss_history

# ============================================================================
# HIERARCHICAL O-INFORMATION COMPUTATION
# ============================================================================

@torch.no_grad()
def estimate_o_information_subset_fast(
    score_net,
    vpsde,
    data,
    subset_indices,
    n_time_steps=10,
    n_samples=1000,
    batch_size=512,
    device='cuda'
):
    """
    Estimate O-information for a specific subset of TFs
    """
    
    score_net.eval()
    score_net = score_net.to(device)
    
    # Sample data
    if len(data) > n_samples:
        sample_idx = np.random.choice(len(data), n_samples, replace=False)
        data_sample = data[sample_idx]
    else:
        data_sample = data
    
    if isinstance(data_sample, np.ndarray):
        data_sample = torch.FloatTensor(data_sample)
    
    data_sample = data_sample.to(device)
    n_samples_actual = len(data_sample)
    
    # Get subset data
    data_subset = data_sample[:, subset_indices]
    n_vars = len(subset_indices)
    
    # Time points
    time_points = torch.linspace(0.01, vpsde.T, n_time_steps, device=device)
    dt = vpsde.T / (n_time_steps - 1)
    
    tc_sum = 0.0
    dtc_sum = 0.0
    
    for t in time_points:
        n_batches = (n_samples_actual + batch_size - 1) // batch_size
        
        tc_step = 0.0
        dtc_step = 0.0
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples_actual)
            batch_data = data_subset[start_idx:end_idx]
            batch_size_curr = len(batch_data)
            
            t_batch = t.repeat(batch_size_curr)
            
            # === Total Correlation ===
            noise_joint = torch.randn_like(batch_data)
            x_joint, _ = vpsde.add_noise(batch_data, t_batch, noise_joint)
            
            # Expand to full dimension for mask
            x_joint_full = torch.zeros(batch_size_curr, data_sample.shape[1], device=device)
            x_joint_full[:, subset_indices] = x_joint
            mask_joint = torch.zeros_like(x_joint_full)
            mask_joint[:, subset_indices] = 1.0
            
            score_joint = score_net(x_joint_full, t_batch, mask_joint)
            score_joint = score_joint[:, subset_indices]
            
            # Marginal scores
            score_marginals_list = []
            
            for i in range(n_vars):
                x_marginal = batch_data.clone()
                noise_i = torch.randn(batch_size_curr, 1, device=device)
                x_i_noised, _ = vpsde.add_noise(
                    batch_data[:, [i]],
                    t_batch,
                    noise_i
                )
                x_marginal[:, i] = x_i_noised.squeeze()
                
                x_marginal_full = torch.zeros(batch_size_curr, data_sample.shape[1], device=device)
                x_marginal_full[:, subset_indices] = x_marginal
                mask_marginal = torch.zeros_like(x_marginal_full)
                mask_marginal[:, subset_indices[i]] = 1.0
                
                score_marginal = score_net(x_marginal_full, t_batch, mask_marginal)
                score_marginals_list.append(score_marginal[:, subset_indices[i]])
            
            score_marginals = torch.stack(score_marginals_list, dim=1)
            
            # TC contribution
            score_diff_tc = score_joint - score_marginals
            tc_step += (score_diff_tc ** 2).sum() / n_samples_actual
            
            # === Dual Total Correlation ===
            score_conditional_list = []
            
            for i in range(n_vars):
                other_indices = [j for j in range(n_vars) if j != i]
                
                x_cond = batch_data.clone()
                
                for j in other_indices:
                    noise_j = torch.randn(batch_size_curr, 1, device=device)
                    x_j_noised, _ = vpsde.add_noise(
                        batch_data[:, [j]],
                        t_batch,
                        noise_j
                    )
                    x_cond[:, j] = x_j_noised.squeeze()
                
                x_cond_full = torch.zeros(batch_size_curr, data_sample.shape[1], device=device)
                x_cond_full[:, subset_indices] = x_cond
                mask_cond = torch.zeros_like(x_cond_full)
                for j in other_indices:
                    mask_cond[:, subset_indices[j]] = 1.0
                
                score_cond = score_net(x_cond_full, t_batch, mask_cond)
                score_conditional_list.append(score_cond[:, subset_indices[i]])
            
            score_conditionals = torch.stack(score_conditional_list, dim=1)
            
            # DTC contribution
            score_diff_dtc = score_joint - score_conditionals
            dtc_step += (score_diff_dtc ** 2).sum() / n_samples_actual
        
        # Weight
        sigma_t = vpsde.sigma(t)
        weight = (sigma_t ** 2) / (1 - sigma_t ** 2 + 1e-8)
        
        tc_sum += weight * tc_step * dt
        dtc_sum += weight * dtc_step * dt
    
    omega = tc_sum - dtc_sum
    
    return {
        'tc': tc_sum.item(),
        'dtc': dtc_sum.item(),
        'omega': omega.item()
    }


def hierarchical_o_information_fast(
    score_net,
    vpsde,
    data,
    tf_names,
    max_order=5,
    top_k_per_order=10,
    n_time_steps=10,
    n_samples=1000,
    batch_size=512,
    device='cuda',
    verbose=True
):
    """
    Hierarchical O-information computation
    Build up from order 3 → 4 → 5
    """
    
    if verbose:
        print(f"\n  Computing hierarchical O-information (orders 3-{max_order})")
    
    n_tfs = len(tf_names)
    hierarchical_results = {}
    
    # ========================================================================
    # ORDER 3: COMPUTE ALL TRIPLETS
    # ========================================================================
    
    if verbose:
        print(f"\n  Order 3: Computing triplets...")
    
    all_triplets = list(combinations(range(n_tfs), 3))
    
    if len(all_triplets) > 5000:
        # Sample if too many
        sample_indices = np.random.choice(len(all_triplets), 5000, replace=False)
        triplets_to_compute = [all_triplets[i] for i in sample_indices]
        if verbose:
            print(f"    Sampling {len(triplets_to_compute)}/{len(all_triplets)} triplets")
    else:
        triplets_to_compute = all_triplets
    
    triplet_results = []
    
    for i, j, k in tqdm(triplets_to_compute, desc="  Triplets", disable=not verbose):
        result = estimate_o_information_subset_fast(
            score_net, vpsde, data, [i, j, k],
            n_time_steps, n_samples, batch_size, device
        )
        
        triplet_results.append({
            'indices': (i, j, k),
            'TF1': tf_names[i],
            'TF2': tf_names[j],
            'TF3': tf_names[k],
            'O_information': result['omega'],
            'TC': result['tc'],
            'DTC': result['dtc']
        })
    
    triplet_df = pd.DataFrame(triplet_results)
    triplet_df = triplet_df.sort_values('O_information', ascending=False)
    
    top_triplets = triplet_df.head(top_k_per_order)
    hierarchical_results[3] = top_triplets
    
    if verbose:
        print(f"    Top triplet: {top_triplets.iloc[0]['TF1']} + {top_triplets.iloc[0]['TF2']} + {top_triplets.iloc[0]['TF3']} | Ω={top_triplets.iloc[0]['O_information']:.4f}")
    
    # ========================================================================
    # BUILD UP HIERARCHICALLY
    # ========================================================================
    
    current_combos = [row['indices'] for _, row in top_triplets.iterrows()]
    
    for order in range(4, max_order + 1):
        if verbose:
            print(f"\n  Order {order}: Extending order-{order-1}...")
        
        next_results = []
        
        for base_combo in tqdm(current_combos, desc=f"  Order {order}", disable=not verbose):
            used_indices = set(base_combo)
            available_indices = [i for i in range(n_tfs) if i not in used_indices]
            
            for new_idx in available_indices:
                extended_combo = tuple(sorted(base_combo + (new_idx,)))
                
                # Check if already computed
                if any(r['indices'] == extended_combo for r in next_results):
                    continue
                
                # Compute O-information
                result = estimate_o_information_subset_fast(
                    score_net, vpsde, data, list(extended_combo),
                    n_time_steps, n_samples, batch_size, device
                )
                
                next_results.append({
                    'indices': extended_combo,
                    'TFs': tuple([tf_names[i] for i in extended_combo]),
                    'O_information': result['omega'],
                    'TC': result['tc'],
                    'DTC': result['dtc']
                })
        
        if len(next_results) == 0:
            if verbose:
                print("    No new combinations found. Stopping.")
            break
        
        # Sort and keep top-k
        next_df = pd.DataFrame(next_results)
        next_df = next_df.sort_values('O_information', ascending=False)
        
        top_next = next_df.head(top_k_per_order)
        hierarchical_results[order] = top_next
        
        if verbose:
            print(f"    Top: {' + '.join(top_next.iloc[0]['TFs'])} | Ω={top_next.iloc[0]['O_information']:.4f}")
        
        # Update for next iteration
        current_combos = [row['indices'] for _, row in top_next.iterrows()]
    
    return hierarchical_results

