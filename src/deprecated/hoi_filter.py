"""
Standalone Hierarchical O-Information Filter
===========================================

Complete implementation with all SOI components embedded.
NO external dependencies on original SOI package!

Only requires: torch, numpy, pandas, networkx, scipy, scanpy, tqdm

Author: Standalone version based on Bounoua et al., 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import issparse
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
import warnings
import math
from copy import deepcopy

import src.methods.tf_filters.base_filter as tf_base
# ============================================================================
# UTILITY FUNCTIONS (from original util.py)
# ============================================================================

def concat_vect(encodings: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate dictionary of tensors into single tensor."""
    return torch.cat(list(encodings.values()), dim=-1)


def deconcat(z: torch.Tensor, var_list: List[str], sizes: List[int]) -> Dict[str, torch.Tensor]:
    """Split concatenated tensor back into dictionary."""
    data = torch.split(z, sizes, dim=1)
    return {var: data[i] for i, var in enumerate(var_list)}


def marginalize_data(x_t: Dict[str, torch.Tensor], mod: str, fill_zeros: bool = False) -> Dict[str, torch.Tensor]:
    """
    Marginalize all variables except 'mod'.
    
    Args:
        x_t: Dictionary of tensors
        mod: Variable to keep
        fill_zeros: If True, fill others with zeros; else with noise
    """
    x = x_t.copy()
    for k in x.keys():
        if k != mod:
            if fill_zeros:
                x[k] = torch.zeros_like(x_t[k])
            else:
                x[k] = torch.randn_like(x_t[k])
    return x


def cond_x_data(x_t: Dict[str, torch.Tensor], data: Dict[str, torch.Tensor], mod: str) -> Dict[str, torch.Tensor]:
    """
    Condition on all variables except 'mod' by using clean data.
    
    Args:
        x_t: Noisy data dictionary
        data: Clean data dictionary
        mod: Variable to keep noisy
    """
    x = x_t.copy()
    for k in x.keys():
        if k != mod:
            x[k] = data[k]
    return x


def expand_mask(mask: torch.Tensor, var_sizes: List[int]) -> torch.Tensor:
    """Expand mask to match variable sizes."""
    return torch.cat([
        mask[:, i].view(mask.shape[0], 1).expand(mask.shape[0], size) 
        for i, size in enumerate(var_sizes)
    ], dim=1)


class EMA(nn.Module):
    """Exponential Moving Average for model weights."""
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), 
                                     model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)


# ============================================================================
# IMPORTANCE SAMPLING (from original importance.py)
# ============================================================================

def sample_vp_truncated_q(shape, beta_min, beta_max, T, t_epsilon=1e-3):
    """Sample time points with importance sampling."""
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    u = torch.rand(*shape).to(T)
    vpsde = VariancePreservingTruncatedSampling(
        beta_min=beta_min, beta_max=beta_max, t_epsilon=t_epsilon
    )
    return vpsde.inv_Phi(u.view(-1), T).view(*shape)


def get_normalizing_constant(shape, T=1.0):
    """Get normalizing constant for importance sampling."""
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    u = torch.rand(*shape).to(T)
    vpsde = VariancePreservingTruncatedSampling(
        beta_min=0.1, beta_max=20.0, t_epsilon=0.001
    )
    return vpsde.normalizing_constant(T=T)


class VariancePreservingTruncatedSampling:
    """Variance Preserving SDE with truncated sampling."""
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20., t_epsilon=1e-3):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.t_epsilon = t_epsilon

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def integral_beta(self, t):
        return 0.5 * t ** 2 * (self.beta_max - self.beta_min) + t * self.beta_min

    def mean_weight(self, t):
        return torch.exp(-0.5 * self.integral_beta(t))

    def var(self, t):
        return 1. - torch.exp(-self.integral_beta(t))

    def std(self, t):
        return self.var(t) ** 0.5

    def g(self, t):
        beta_t = self.beta(t)
        return beta_t ** 0.5

    def r(self, t):
        return self.beta(t) / self.var(t)

    def t_new(self, t):
        mask_le_t_eps = (t <= self.t_epsilon).float()
        t_new = mask_le_t_eps * self.t_epsilon + (1. - mask_le_t_eps) * t
        return t_new

    def unpdf(self, t):
        t_new = self.t_new(t)
        unprob = self.r(t_new)
        return unprob

    def antiderivative(self, t):
        return torch.log(1. - torch.exp(-self.integral_beta(t))) + self.integral_beta(t)

    def phi_t_le_t_eps(self, t):
        t_eps = torch.tensor(float(self.t_epsilon))
        return self.r(t_eps).item() * t

    def phi_t_gt_t_eps(self, t):
        t_eps = torch.tensor(float(self.t_epsilon))
        return self.phi_t_le_t_eps(t_eps).item() + self.antiderivative(t) - self.antiderivative(t_eps).item()

    def normalizing_constant(self, T):
        return self.phi_t_gt_t_eps(T)

    def pdf(self, t, T):
        Z = self.normalizing_constant(T)
        prob = self.unpdf(t) / Z
        return prob

    def Phi(self, t, T):
        Z = self.normalizing_constant(T)
        t_new = self.t_new(t)
        mask_le_t_eps = (t <= self.t_epsilon).float()
        phi = mask_le_t_eps * self.phi_t_le_t_eps(t) + (1. - mask_le_t_eps) * self.phi_t_gt_t_eps(t_new)
        return phi / Z

    def inv_Phi(self, u, T):
        t_eps = torch.tensor(float(self.t_epsilon))
        Z = self.normalizing_constant(T)
        r_t_eps = self.r(t_eps).item()
        antdrv_t_eps = self.antiderivative(t_eps).item()
        mask_le_u_eps = (u <= self.t_epsilon * r_t_eps / Z).float()
        a = self.beta_max - self.beta_min
        b = self.beta_min
        inv_phi = mask_le_u_eps * Z / r_t_eps * u + (1. - mask_le_u_eps) * \
                  (-b + (b ** 2 + 2. * a * torch.log(
                      1. + torch.exp(Z * u + antdrv_t_eps - r_t_eps * self.t_epsilon))) ** 0.5) / a
        return inv_phi


# ============================================================================
# VP-SDE IMPLEMENTATION (from original SDE.py)
# ============================================================================

class VP_SDE:
    """Variance Preserving SDE for score matching."""
    
    def __init__(
        self,
        beta_min=0.1,
        beta_max=20,
        N=1000,
        importance_sampling=True,
        o_inf_order=1,
        weight_s_functions=True,
        marginals=1,
        var_sizes=[1, 1, 1]
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.marginals = marginals
        self.var_sizes = var_sizes
        self.rand_batch = True
        self.N = N
        self.T = 1
        self.importance_sampling = importance_sampling
        self.nb_var = len(self.var_sizes)
        self.weight_s_functions = weight_s_functions
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.masks = self.get_masks_training(o_inf_order=o_inf_order)

    def set_device(self, device):
        self.device = device
        self.masks = self.masks.to(device)

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, t):
        """Returns drift and diffusion coefficient."""
        return -0.5 * self.beta_t(t), torch.sqrt(self.beta_t(t))

    def marg_prob(self, t, x):
        """Returns mean and std of marginal distribution P_t(x_t)."""
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        return mean.view(-1, 1) * torch.ones_like(x, device=self.device), \
               std.view(-1, 1) * torch.ones_like(x, device=self.device)

    def sample(self, x_0, t):
        """Sample from P(x_t | x_0) - forward SDE."""
        mean, std = self.marg_prob(t, t)
        z = torch.randn_like(x_0, device=self.device)
        x_t = x_0 * mean + std * z
        return x_t, z, mean, std

    def train_step(self, data, score_net, eps=1e-5):
        """Single training step for score network."""
        x_0 = concat_vect(data)
        bs = x_0.size(0)

        if self.importance_sampling:
            t = self.sample_importance_sampling_t(shape=(x_0.shape[0], 1)).to(self.device)
        else:
            t = ((self.T - eps) * torch.rand((x_0.shape[0], 1)) + eps).to(self.device)

        # Randomly sample a mask
        if self.rand_batch:
            i = torch.randint(low=1, high=len(self.masks) + 1, size=(bs,)) - 1
        else:
            i = (torch.randint(low=1, high=len(self.masks) + 1, size=(1,)) - 1).expand(bs)

        mask = self.masks[i.long(), :]
        mask_data = expand_mask(mask, self.var_sizes)
        mask_data_marg = (mask_data < 0).float()
        mask_data_diffused = mask_data.clip(0, 1)

        x_t, Z, _, _ = self.sample(x_0=x_0, t=t)
        x_t = mask_data_diffused * x_t + (1 - mask_data_diffused) * x_0
        x_t = x_t * (1 - mask_data_marg)
        
        if self.marginals == 1:
            x_t = x_t + mask_data_marg * torch.randn_like(x_0, device=self.device)

        score = score_net(x_t, t=t, mask=mask, std=None) * mask_data_diffused
        Z = Z * mask_data_diffused

        total_size = score.size(1)
        weight = (((total_size - torch.sum(mask_data_diffused, dim=1)) / total_size) + 1).view(bs, 1)
        loss = (weight * torch.square(score - Z)).sum(1, keepdim=False)
        
        return loss

    def get_masks_training(self, o_inf_order):
        """Generate training masks for different score functions."""
        nb_var = self.nb_var
        masks = [list(i) for i in __import__('itertools').product([0, 1, -1], repeat=nb_var)]
        
        if o_inf_order == 1:
            masks = [s for s in masks if 
                     (sum(s) == nb_var) or  # Joint
                     (np.sum(np.array(s) == 1) and np.sum(np.array(s) == -1) == nb_var - 1) or  # Marginal
                     (np.sum(np.array(s) == 1) == 1 and np.min(np.array(s)) == 0)]  # Conditional
        
        if self.weight_s_functions:
            return self.weight_masks(masks)
        else:
            np.random.shuffle(masks)
            return torch.tensor(masks, device=self.device)

    def weight_masks(self, masks):
        """Weight masks so complex score functions are picked more often."""
        masks_w = []
        for s in masks:
            nb_var_inset = np.sum(np.array(s) == 1) + np.sum(np.array(s) == 0) // 2
            for i in range(nb_var_inset):
                masks_w.append(s)
        np.random.shuffle(masks_w)
        return torch.tensor(masks_w, device=self.device)

    def sample_importance_sampling_t(self, shape):
        """Sample time with importance sampling."""
        return sample_vp_truncated_q(shape, self.beta_min, self.beta_max, T=self.T)


# ============================================================================
# INFORMATION MEASURES (from original info_measures.py)
# ============================================================================

def get_tc(sde, s_joint, s_marg, g, importance_sampling):
    """
    Calculate Total Correlation using score functions.
    
    TC = ∫ 0.5 * g² * ||S_joint - S_marg||² dt
    """
    M = g.shape[0]
    
    if importance_sampling:
        const = get_normalizing_constant((1,)).to(sde.device)
        tc = const * 0.5 * ((concat_vect(s_joint) - concat_vect(s_marg)) ** 2).sum() / M
    else:
        tc = 0.5 * (g ** 2 * (concat_vect(s_joint) - concat_vect(s_marg)) ** 2).sum() / M
    
    return tc.item()


def get_dtc(sde, s_joint, s_cond, g, importance_sampling):
    """
    Calculate Dual Total Correlation using score functions.
    
    DTC = ∫ 0.5 * g² * ||S_joint - S_cond||² dt
    """
    M = g.shape[0]
    
    if importance_sampling:
        const = get_normalizing_constant((1,)).to(sde.device)
        dtc = const * 0.5 * ((concat_vect(s_joint) - concat_vect(s_cond)) ** 2).sum() / M
    else:
        dtc = 0.5 * (g ** 2 * (concat_vect(s_joint) - concat_vect(s_cond)) ** 2).sum() / M
    
    return dtc.item()


def compute_all_measures(tc, dtc):
    """Given TC and DTC, compute all information measures."""
    return {
        "tc": tc,
        "dtc": dtc,
        "o_inf": tc - dtc,
        "s_inf": tc + dtc,
    }


# ============================================================================
# TRANSFORMER MODEL (simplified from original transformer.py)
# ============================================================================

def modulate(x, shift, scale):
    """Modulation for adaptive layer norm."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sinusoidal positional embeddings."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class DiTBlock(nn.Module):
    """DiT block with adaptive layer norm zero conditioning."""
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final layer of DiT."""
    
    def __init__(self, hidden_size, out_size, nb_var, encoding=False):
        super().__init__()
        self.encoding = encoding
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, out_size, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class VarEmbed(nn.Module):
    """Variable embedding layer."""
    
    def __init__(self, sizes, embed_dim, norm_layer=None, bias=True):
        super().__init__()
        self.sizes = sizes
        norm_l = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.norm = nn.ModuleList([norm_l for _ in sizes])
        self.proj = nn.ModuleList([nn.Linear(size, embed_dim, bias=bias) for size in sizes])

    def forward(self, x_var):
        i = np.arange(len(x_var))
        proj = np.array(self.proj, dtype=object)[i]
        norm = np.array(self.norm, dtype=object)[i]
        
        x = [proj_x(x_var[idx].float()) for idx, proj_x in enumerate(proj)]
        x = [norm_x(x[idx]) for idx, norm_x in enumerate(norm)]
        return torch.stack(x).permute(1, 0, 2)


class VarDeEmbed(nn.Module):
    """Variable de-embedding layer."""
    
    def __init__(self, sizes, embed_dim, norm_layer=None, bias=True):
        super().__init__()
        self.sizes = sizes
        self.proj = nn.ModuleList([nn.Linear(embed_dim, size, bias=bias) for size in sizes])

    def forward(self, x_var, i=[]):
        x_var = x_var.permute(1, 0, 2)
        if len(i) == 0:
            i = np.arange(len(x_var))
        proj = np.array(self.proj, dtype=object)[i]
        x = [proj_x(x_var[idx]) for idx, proj_x in enumerate(proj)]
        return x


class DiT_Enc(nn.Module):
    """DiT encoder."""
    
    def __init__(self, hidden_size, var_sizes, depth, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, hidden_size, nb_var=len(var_sizes), encoding=True)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t):
        c = t
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return x


class DiT(nn.Module):
    """Diffusion Transformer for score estimation."""
    
    def __init__(self, hidden_size=128, depth=4, num_heads=4, mlp_ratio=4.0, var_list=None):
        super().__init__()
        self.num_heads = num_heads
        self.var_sizes = list(var_list.values())
        self.var_list = var_list
        self.hidden_size = hidden_size

        self.var_enc = DiT_Enc(
            hidden_size=hidden_size,
            var_sizes=self.var_sizes,
            depth=depth // 2,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )

        self.x_embedder = VarEmbed(
            sizes=self.var_sizes,
            embed_dim=hidden_size,
            norm_layer=nn.LayerNorm
        )
        
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(
            hidden_size, hidden_size, encoding=False, nb_var=len(self.var_sizes)
        )
        
        self.unembed_var = VarDeEmbed(
            sizes=self.var_sizes,
            embed_dim=hidden_size,
            norm_layer=None
        )
        
        self.pos_embed = self.get_pos_embed(np.array(np.arange(len(self.var_sizes))))

    def get_pos_embed(self, i):
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, i)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)

    def forward(self, x, t=None, mask=None, std=None):
        x = torch.split(x, self.var_sizes, dim=1)
        x_all = self.x_embedder(x) + self.pos_embed.to(x[0].device)
        
        t = self.t_embedder(t).squeeze()
        
        mask = mask.view(mask.shape[0], mask.shape[1], 1)
        x, x_c = (mask > 0).int() * x_all, (mask == 0).int() * x_all
        
        mask_cond = ((mask == 0).sum(dim=1) > 0).int().view(mask.shape[0], 1)
        y = self.var_enc(x_c, t=torch.zeros_like(t))
        y = y.sum(dim=1) * mask_cond
        
        c = t + y

        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unembed_var(x)
        
        out = torch.cat(x, dim=1)
        if std is not None:
            return out / std
        else:
            return out


# ============================================================================
# HIERARCHICAL O-INFORMATION FILTER
# ============================================================================



class HierarchicalOInformationFilter(tf_base.BaseTFIdentityPipeline):
    """
    Standalone Hierarchical O-information filter.
    NO external dependencies on original SOI package!
    """
    
    def __init__(
        self,
        adata,
        tf_list: List[str],
        target_cell_type: str,
        cell_type_key: str = 'cell_type',
        scgx_sig_file: str = None,
        chipseq_file: str = None,
        known_identity_tfs: List[str] = None,
        verbose: bool = True,
        # O-information parameters
        top_jsd_n: int = 30,
        max_order: int = 3,
        hidden_dim: int = 128,
        training_epochs: int = 100,
        batch_size: int = 512,
        n_mc_samples: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        super().__init__(
            adata=adata,
            tf_list=tf_list,
            target_cell_type=target_cell_type,
            cell_type_key=cell_type_key,
            scgx_sig_file=scgx_sig_file,
            chipseq_file=chipseq_file,
            known_identity_tfs=known_identity_tfs,
            verbose=verbose,
            **kwargs
        )
        
        self.top_jsd_n = top_jsd_n
        self.max_order = max_order
        self.hidden_dim = hidden_dim
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.n_mc_samples = n_mc_samples
        self.device = torch.device(device)
        
        self.score_net = None
        self.sde = None
        self.selected_genes = None
        
        self.oi_results = {
            'top_jsd_genes': None,
            'triplet_scores': None,
            'top_triplets': None,
            'training_time': None,
            'inference_time': None
        }
    
    def apply_additional_filters(self, core_tfs: List[str]) -> List[str]:
        """Apply O-information based filtering."""
        if self.verbose:
            print(f"\n[O-Information Analysis]")
            print(f"  Input TFs: {len(core_tfs)}")
        
        # Filter to top N by JSD
        top_jsd_genes = self._filter_by_jsd(core_tfs)
        self.oi_results['top_jsd_genes'] = top_jsd_genes
        
        if len(top_jsd_genes) < 3:
            warnings.warn("Too few genes for O-information analysis")
            return top_jsd_genes
        
        # Train SOI network
        if self.verbose:
            print(f"\n  Training SOI score network on {len(top_jsd_genes)} genes...")
        
        train_start = time.time()
        self._train_soi_network(top_jsd_genes)
        self.oi_results['training_time'] = time.time() - train_start
        
        if self.verbose:
            print(f"    Training completed in {self.oi_results['training_time']:.1f}s")
        
        # Compute O-information
        if self.verbose:
            n_triplets = len(list(combinations(top_jsd_genes, 3)))
            print(f"\n  Computing O-information for {n_triplets} triplets...")
        
        infer_start = time.time()
        triplet_results = self._compute_oi_all_triplets(top_jsd_genes)
        self.oi_results['inference_time'] = time.time() - infer_start
        
        if self.verbose:
            print(f"    Inference completed in {self.oi_results['inference_time']:.1f}s")
        
        # Select top triplets
        top_triplets = self._select_top_triplets(triplet_results)
        self.oi_results['triplet_scores'] = triplet_results
        self.oi_results['top_triplets'] = top_triplets
        
        selected_tfs = self._extract_genes_from_triplets(top_triplets)
        
        if self.verbose:
            print(f"\n  Selected {len(selected_tfs)} TFs from top synergistic triplets")
            if len(top_triplets) > 0:
                print(f"    Top synergistic triplet: {top_triplets.iloc[0]['genes']}")
                print(f"    Ω = {top_triplets.iloc[0]['omega']:.4f}")
        
        return selected_tfs
    
    def _filter_by_jsd(self, tfs: List[str]) -> List[str]:
        """Filter to top N genes - simplified version."""
        if self.verbose:
            print(f"  Filtering to top {self.top_jsd_n} genes...")
        
        # Simple variance-based filtering if JSD not available
        gene_vars = []
        for tf in tfs:
            if tf in self.adata.var_names:
                X = self.adata_target[:, tf].X
                if issparse(X):
                    X = X.toarray()
                var = np.var(X)
                gene_vars.append((tf, var))
        
        gene_vars.sort(key=lambda x: x[1], reverse=True)
        top_genes = [gene for gene, _ in gene_vars[:self.top_jsd_n]]
        
        if self.verbose:
            print(f"    Selected {len(top_genes)} genes")
        
        return top_genes
    
    def _train_soi_network(self, gene_list: List[str]):
        """Train SOI score network."""
        # Ensure gene_list is a proper list of strings (not pandas Index, etc.)
        self.selected_genes = list(gene_list) if not isinstance(gene_list, list) else gene_list
        n_genes = len(self.selected_genes)
        
        # Extract and normalize data - use self.selected_genes consistently
        X = self.adata_target[:, self.selected_genes].X
        if issparse(X):
            X = X.toarray()
        
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        X = torch.FloatTensor(X).to(self.device)
        
        # Initialize network - use self.selected_genes consistently
        # Using OrderedDict to be explicit about order
        from collections import OrderedDict
        var_list = OrderedDict((gene, 1) for gene in self.selected_genes)
        
        self.score_net = DiT(
            depth=4,
            hidden_size=self.hidden_dim,
            var_list=var_list
        ).to(self.device)
        
        # Initialize SDE
        self.sde = VP_SDE(
            importance_sampling=True,
            var_sizes=[1] * n_genes,
            weight_s_functions=True,
            o_inf_order=1,
            marginals=-1
        )
        self.sde.set_device(self.device)
        
        # Create dataset - use self.selected_genes consistently
        dataset = self._create_torch_dataset(X, self.selected_genes)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # Train
        optimizer = torch.optim.Adam(self.score_net.parameters(), lr=1e-3)
        
        self.score_net.train()
        for epoch in range(self.training_epochs):
            epoch_loss = 0
            n_batches = 0
            
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.sde.train_step(batch, self.score_net).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if self.verbose and epoch % 20 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"      Epoch {epoch}/{self.training_epochs}, Loss: {avg_loss:.4f}")
        
        self.score_net.eval()
    
    def _create_torch_dataset(self, X: torch.Tensor, gene_list: List[str]):
        """Create PyTorch dataset."""
        class GeneDataset(torch.utils.data.Dataset):
            def __init__(self, data, genes):
                self.data = data
                self.genes = genes
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                sample = {}
                for i, gene in enumerate(self.genes):
                    sample[gene] = self.data[idx, i:i+1]
                return sample
        
        return GeneDataset(X, gene_list)
    
    def _compute_oi_all_triplets(self, gene_list: List[str]) -> pd.DataFrame:
        """Compute O-information for all triplets."""
        all_triplets = list(combinations(gene_list, 3))
        results = []
        
        if self.verbose:
            pbar = tqdm(all_triplets, desc="Computing O-info")
        else:
            pbar = all_triplets
        
        for triplet in pbar:
            triplet = tuple(sorted(triplet))
            try:
                oi_dict = self._compute_oi_single_triplet(triplet)
                results.append({
                    'genes': triplet,
                    'omega': oi_dict['o_inf'],
                    'tc': oi_dict['tc'],
                    'dtc': oi_dict['dtc'],
                    's_inf': oi_dict['s_inf']
                })
            except Exception as e:
                if self.verbose:
                    print(f"      Warning: Failed for {triplet}: {type(e).__name__}: {str(e)}")
                continue
        
        df = pd.DataFrame(results)
        df = df.sort_values('omega', ascending=False)
        return df
    
    def _compute_oi_single_triplet(self, triplet: Tuple[str, str, str]) -> Dict[str, float]:
        """
        Compute O-information for single triplet.
        
        CRITICAL FIX: Must use ALL selected genes (not just triplet) because
        the score network was trained on all selected genes and expects that dimensionality.
        We use masking to focus computation on the triplet.
        """
        # DEBUG: Validate selected_genes type
        if not isinstance(self.selected_genes, (list, tuple)):
            raise TypeError(f"self.selected_genes must be list/tuple, got {type(self.selected_genes)}")
        
        # Ensure selected_genes contains only strings
        if not all(isinstance(g, str) for g in self.selected_genes):
            raise TypeError(f"self.selected_genes must contain only strings, got types: {[type(g) for g in self.selected_genes]}")
        
        # Extract ALL selected genes (not just triplet!)
        X = self.adata_target[:, self.selected_genes].X
        if issparse(X):
            X = X.toarray()
        
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Create data dict for ALL genes
        all_genes_data = {}
        for i, gene in enumerate(self.selected_genes):
            tensor = torch.FloatTensor(X[:, i:i+1]).to(self.device)
            all_genes_data[gene] = tensor
            # DEBUG: Validate tensor type
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"all_genes_data[{gene}] is {type(tensor)}, not Tensor!")
        
        x_0 = concat_vect(all_genes_data)
        M = len(x_0)
        
        # Get indices of triplet genes within all selected genes
        n_genes = len(self.selected_genes)
        
        # Validate triplet genes are in selected_genes
        for gene in triplet:
            if gene not in self.selected_genes:
                raise ValueError(f"Gene {gene} in triplet not found in selected_genes")
        
        triplet_indices = [self.selected_genes.index(g) for g in triplet]
        
        # Create masks for the full gene set (focusing on triplet)
        marg_masks = {}
        cond_masks = {}
        for gene in triplet:
            idx = self.selected_genes.index(gene)
            
            # Marginal mask: -1 everywhere except +1 at target position
            mask_m = [-1] * n_genes
            mask_m[idx] = 1
            marg_masks[gene] = torch.FloatTensor(mask_m).to(self.device)
            
            # Conditional mask: 0 everywhere except +1 at target position
            mask_c = [0] * n_genes
            mask_c[idx] = 1
            cond_masks[gene] = torch.FloatTensor(mask_c).to(self.device)
        
        # Sizes for deconcat: one per gene
        sizes = [1] * n_genes
        
        tc_samples = []
        dtc_samples = []
        
        self.score_net.eval()
        with torch.no_grad():
            for mc_iter in range(self.n_mc_samples):
                t = self.sde.sample_importance_sampling_t(shape=(M, 1)).to(self.device)
                # CRITICAL: g is the diffusion coefficient tensor - do not overwrite!
                _, g = self.sde.sde(t)
                x_t, _, _, std = self.sde.sample(x_0=x_0, t=t)
                
                # DEBUG: Validate x_t is tensor
                if not isinstance(x_t, torch.Tensor):
                    raise TypeError(f"x_t is {type(x_t)}, not Tensor!")
                
                # Deconcat using ALL genes
                X_t = deconcat(x_t, self.selected_genes, sizes)
                
                # DEBUG: Validate X_t contents
                if not isinstance(X_t, dict):
                    raise TypeError(f"X_t is {type(X_t)}, not dict!")
                
                for key, val in X_t.items():
                    if not isinstance(key, str):
                        raise TypeError(f"X_t key {key} is {type(key)}, not str!")
                    if not isinstance(val, torch.Tensor):
                        raise TypeError(f"X_t[{key}] is {type(val)}, not Tensor! X_t.keys()={list(X_t.keys())}")
                
                # Joint score (mask shows which genes are in the triplet)
                mask_joint = torch.zeros(n_genes).to(self.device)
                for idx in triplet_indices:
                    mask_joint[idx] = 1
                    
                s_joint_concat = -self.score_net(
                    x_t, t=t,
                    mask=mask_joint.unsqueeze(0).expand(M, -1),
                    std=None
                )
                
                # DEBUG: Validate score output
                if not isinstance(s_joint_concat, torch.Tensor):
                    raise TypeError(f"s_joint_concat is {type(s_joint_concat)}, not Tensor!")
                
                s_joint_full = deconcat(s_joint_concat, self.selected_genes, sizes)
                
                # DEBUG: Validate s_joint_full
                for key, val in s_joint_full.items():
                    if not isinstance(val, torch.Tensor):
                        raise TypeError(f"s_joint_full[{key}] is {type(val)}, not Tensor!")
                
                # Extract only triplet genes IN THE SAME ORDER
                s_joint = {gene_name: s_joint_full[gene_name] for gene_name in triplet}
                
                # Marginal scores - build with same key order
                s_marg = {}
                for gene in triplet:  # Iterate in triplet order
                    # Marginalize: keep only target gene, zero out all others
                    x_marg = {}
                    for gene_name in self.selected_genes:
                        # DEBUG: Check X_t[gene_name] type before using it
                        if not isinstance(X_t[gene_name], torch.Tensor):
                            raise TypeError(f"Before zeros_like: X_t[{gene_name}] is {type(X_t[gene_name])}, not Tensor!")
                        
                        if gene_name == gene:
                            x_marg[gene_name] = X_t[gene_name]
                        else:
                            x_marg[gene_name] = torch.zeros_like(X_t[gene_name])
                    x_marg_concat = concat_vect(x_marg)
                    
                    s_marg_concat = -self.score_net(
                        x_marg_concat, t=t,
                        mask=marg_masks[gene].unsqueeze(0).expand(M, -1),
                        std=None
                    )
                    s_marg_full = deconcat(s_marg_concat, self.selected_genes, sizes)
                    s_marg[gene] = s_marg_full[gene]
                
                # Ensure s_marg has the same key order as triplet
                s_marg = {gene: s_marg[gene] for gene in triplet}
                
                # Conditional scores - build with same key order
                s_cond = {}
                for gene in triplet:  # Iterate in triplet order
                    # Condition: noisy target, clean data for other triplet genes, zeros elsewhere
                    x_cond = {}
                    for gene_name in self.selected_genes:
                        if gene_name == gene:
                            x_cond[gene_name] = X_t[gene_name]  # Noisy target
                        elif gene_name in triplet:
                            x_cond[gene_name] = all_genes_data[gene_name]  # Clean conditioning from triplet
                        else:
                            x_cond[gene_name] = torch.zeros_like(X_t[gene_name])  # Zero out non-triplet
                    x_cond_concat = concat_vect(x_cond)
                    
                    s_cond_concat = -self.score_net(
                        x_cond_concat, t=t,
                        mask=cond_masks[gene].unsqueeze(0).expand(M, -1),
                        std=None
                    )
                    s_cond_full = deconcat(s_cond_concat, self.selected_genes, sizes)
                    s_cond[gene] = s_cond_full[gene]
                
                # Ensure s_cond has the same key order as triplet
                s_cond = {gene: s_cond[gene] for gene in triplet}
                
                # Validate score dictionaries before O-information calculation
                if len(s_joint) != 3:
                    raise ValueError(f"s_joint should have 3 genes, has {len(s_joint)}: {list(s_joint.keys())}")
                if len(s_marg) != 3:
                    raise ValueError(f"s_marg should have 3 genes, has {len(s_marg)}: {list(s_marg.keys())}")
                if len(s_cond) != 3:
                    raise ValueError(f"s_cond should have 3 genes, has {len(s_cond)}: {list(s_cond.keys())}")
                
                for gene_name in triplet:
                    if gene_name not in s_joint:
                        raise KeyError(f"Gene {gene_name} missing from s_joint")
                    if gene_name not in s_marg:
                        raise KeyError(f"Gene {gene_name} missing from s_marg")
                    if gene_name not in s_cond:
                        raise KeyError(f"Gene {gene_name} missing from s_cond")
                    if not isinstance(s_joint[gene_name], torch.Tensor):
                        raise TypeError(f"s_joint[{gene_name}] is {type(s_joint[gene_name])}, not Tensor!")
                    if not isinstance(s_marg[gene_name], torch.Tensor):
                        raise TypeError(f"s_marg[{gene_name}] is {type(s_marg[gene_name])}, not Tensor!")
                    if not isinstance(s_cond[gene_name], torch.Tensor):
                        raise TypeError(f"s_cond[{gene_name}] is {type(s_cond[gene_name])}, not Tensor!")
                
                # Validate g is tensor with correct shape
                if not isinstance(g, torch.Tensor):
                    raise TypeError(f"g is {type(g)}, not Tensor!")
                if g.ndim != 2 or g.shape[1] != 1:
                    raise ValueError(f"g should have shape (M, 1), got {g.shape}")
                
                # Validate concatenated score shapes match
                s_joint_cat = concat_vect(s_joint)
                s_marg_cat = concat_vect(s_marg)
                s_cond_cat = concat_vect(s_cond)
                
                if s_joint_cat.shape != s_marg_cat.shape:
                    raise ValueError(f"Shape mismatch: s_joint {s_joint_cat.shape} vs s_marg {s_marg_cat.shape}")
                if s_joint_cat.shape != s_cond_cat.shape:
                    raise ValueError(f"Shape mismatch: s_joint {s_joint_cat.shape} vs s_cond {s_cond_cat.shape}")
                if s_joint_cat.shape[0] != g.shape[0]:
                    raise ValueError(f"Batch size mismatch: scores {s_joint_cat.shape[0]} vs g {g.shape[0]}")
                
                tc = get_tc(self.sde, s_joint=s_joint, s_marg=s_marg, g=g, importance_sampling=True)
                dtc = get_dtc(self.sde, s_joint=s_joint, s_cond=s_cond, g=g, importance_sampling=True)
                
                tc_samples.append(tc)
                dtc_samples.append(dtc)
        
        tc_mean = np.mean(tc_samples)
        dtc_mean = np.mean(dtc_samples)
        
        return compute_all_measures(tc_mean, dtc_mean)
    
    def _get_masks_for_triplet(self, var_list: List[str]) -> Tuple[Dict, Dict]:
        """Create proper masks."""
        n_vars = len(var_list)
        marg_masks = {}
        cond_masks = {}
        
        for i, gene in enumerate(var_list):
            mask_m = [-1] * n_vars
            mask_m[i] = 1
            marg_masks[gene] = torch.FloatTensor(mask_m).to(self.device)
            
            mask_c = [0] * n_vars
            mask_c[i] = 1
            cond_masks[gene] = torch.FloatTensor(mask_c).to(self.device)
        
        return marg_masks, cond_masks
    
    def _marginalize_dict(self, X_t: Dict, keep_var: str, fill_zeros: bool = True) -> Dict:
        """Marginalize all variables except keep_var."""
        X_marg = {}
        for var in X_t.keys():
            if var == keep_var:
                X_marg[var] = X_t[var]
            else:
                X_marg[var] = torch.zeros_like(X_t[var]) if fill_zeros else torch.randn_like(X_t[var])
        return X_marg
    
    def _condition_dict(self, X_t: Dict, target_var: str, clean_data: Dict) -> Dict:
        """Condition on other variables."""
        X_cond = {}
        for var in X_t.keys():
            X_cond[var] = X_t[var] if var == target_var else clean_data[var]
        return X_cond
    
    def _select_top_triplets(self, triplet_df: pd.DataFrame, top_pct: float = 0.1) -> pd.DataFrame:
        """Select top triplets."""
        n_top = max(1, int(len(triplet_df) * top_pct))
        return triplet_df.head(n_top)
    
    def _extract_genes_from_triplets(self, triplet_df: pd.DataFrame) -> List[str]:
        """Extract genes from top triplets."""
        gene_counts = {}
        for _, row in triplet_df.iterrows():
            for gene in row['genes']:
                gene_counts[gene] = gene_counts.get(gene, 0) + 1
        
        sorted_genes = sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)
        return [gene for gene, _ in sorted_genes]
    
    def build_network(self, filtered_tfs: List[str]) -> nx.Graph:
        """Build network from O-information."""
        if self.verbose:
            print(f"\n  Building O-information network...")
        
        G = nx.Graph()
        G.add_nodes_from(filtered_tfs)
        
        if self.oi_results['triplet_scores'] is None:
            return G
        
        triplet_df = self.oi_results['triplet_scores']
        synergistic = triplet_df[triplet_df['omega'] > 0]
        
        for _, row in synergistic.iterrows():
            genes = row['genes']
            omega = row['omega']
            
            if all(g in filtered_tfs for g in genes):
                for i in range(3):
                    for j in range(i+1, 3):
                        g1, g2 = genes[i], genes[j]
                        if G.has_edge(g1, g2):
                            G[g1][g2]['weight'] = max(G[g1][g2]['weight'], omega)
                        else:
                            G.add_edge(g1, g2, weight=omega)
        
        if self.verbose:
            print(f"    Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def identify_key_tfs(self, graph: nx.Graph, filtered_tfs: List[str]) -> List[str]:
        """Identify key TFs."""
        if graph is None or graph.number_of_edges() == 0:
            return filtered_tfs[:min(15, len(filtered_tfs))]
        
        centrality = {}
        for node in graph.nodes():
            weight_sum = sum([graph[node][neighbor].get('weight', 1.0)
                            for neighbor in graph.neighbors(node)])
            centrality[node] = weight_sum
        
        sorted_tfs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        n_select = min(15, len(sorted_tfs))
        key_tfs = [tf for tf, _ in sorted_tfs[:n_select]]
        
        if self.verbose:
            print(f"\n  Selected {len(key_tfs)} key TFs by O-information centrality")
        
        return key_tfs
    
    def get_summary(self) -> Dict:
        """Get summary."""
        summary = {
            'oi_analysis': {
                'top_jsd_genes': len(self.oi_results.get('top_jsd_genes', [])),
                'n_triplets_computed': len(self.oi_results.get('triplet_scores', [])),
                'training_time_sec': self.oi_results.get('training_time', 0),
                'inference_time_sec': self.oi_results.get('inference_time', 0),
            }
        }
        
        if self.oi_results.get('top_triplets') is not None and len(self.oi_results['top_triplets']) > 0:
            top_trip = self.oi_results['top_triplets'].iloc[0]
            summary['oi_analysis']['top_triplet'] = {
                'genes': top_trip['genes'],
                'omega': top_trip['omega'],
                'tc': top_trip['tc'],
                'dtc': top_trip['dtc']
            }
        
        return summary