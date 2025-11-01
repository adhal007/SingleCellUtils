"""
Fast O-Information Filter with Gradient-Based Higher-Order Search
==================================================================

Complete implementation in a single file.

Features:
- Fast continuous O-information computation (Gaussian or KNN)
- Exhaustive triplet search on top JSD genes
- Gradient-based higher-order search (PSO/SA)
- Fixed DTC calculation and sign convention

Author: Combined implementation
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Set
import networkx as nx
from tqdm import tqdm
import warnings
from importlib import reload
import src.methods.tf_filters.base_filter as tf_base
reload(tf_base)

# ============================================================================
# O-INFORMATION COMPUTATION
# ============================================================================

class GaussianOInformation:
    """
    Fast O-information estimation using Gaussian approximation.
    
    For multivariate Gaussian: O(X1, ..., Xn) = TC - DTC
    where TC and DTC can be computed from correlation matrices.
    """
    
    def __init__(self, method='gaussian'):
        """
        Args:
            method: 'gaussian' (fast, assumes normal) or 'knn' (slower, non-parametric)
        """
        self.method = method
        
    def compute_oi(self, X: np.ndarray) -> Dict[str, float]:
        """
        Compute O-information for multivariate data.
        
        Args:
            X: (n_samples, n_variables) array
            
        Returns:
            dict with 'omega', 'tc', 'dtc', 's_inf', 'synergy', 'redundancy'
        """
        if self.method == 'gaussian':
            return self._gaussian_oi(X)
        elif self.method == 'knn':
            return self._knn_oi(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _gaussian_oi(self, X: np.ndarray) -> Dict[str, float]:
        """
        CORRECTED Gaussian O-information using COVARIANCE (not correlation).
        
        For Gaussian X with covariance Σ:
        h(X) = (n/2)*log(2πe) + (1/2)*log|Σ|
        
        TC = Σᵢ h(Xᵢ) - h(X)
        DTC = Σᵢ h(X₋ᵢ) - (n-1)*h(X)
        Ω = TC - DTC
        """
        n_samples, n_vars = X.shape
        
        if n_samples < n_vars:
            warnings.warn(
                f"Few samples ({n_samples}) vs variables ({n_vars}). "
                f"Results may be unreliable."
            )
        
        # Center the data (DON'T standardize - we need real variances!)
        # X_centered = X - X.mean(axis=0)
        X_centered = X
        # ============================================================
        # KEY FIX: Use COVARIANCE matrix, not correlation!
        # ============================================================
        Sigma = np.cov(X_centered, rowvar=False)
        
        # Add small regularization for numerical stability
        Sigma = Sigma + np.eye(n_vars) * 1e-6
        
        # Check for valid covariance
        if np.any(np.diag(Sigma) <= 0):
            warnings.warn("Zero or negative variances detected!")
            Sigma = Sigma + np.eye(n_vars) * 1e-3
        
        # ============================================================
        # Joint entropy: h(X) = (n/2)*log(2πe) + (1/2)*log|Σ|
        # ============================================================
        sign, logdet_joint = np.linalg.slogdet(Sigma)
        if sign <= 0:
            warnings.warn("Covariance matrix not positive definite!")
            # Fallback to absolute value
            logdet_joint = np.log(np.abs(np.linalg.det(Sigma)) + 1e-10)
        
        # Include the (n/2)*log(2πe) constant!
        const_term = 0.5 * n_vars * np.log(2 * np.pi * np.e)
        H_joint = 0.5 * logdet_joint + const_term
        
        # ============================================================
        # Marginal entropies: h(Xᵢ) = 0.5*log(2πe*σᵢ²)
        # ============================================================
        H_marginals = np.zeros(n_vars)
        for i in range(n_vars):
            var_i = Sigma[i, i]
            if var_i <= 0:
                warnings.warn(f"Non-positive variance for variable {i}: {var_i}")
                var_i = 1e-10
            # h(Xᵢ) = 0.5 * log(2πe * σᵢ²)
            H_marginals[i] = 0.5 * np.log(2 * np.pi * np.e * var_i)
        
        # ============================================================
        # Subset entropies: h(X₋ᵢ) for each variable left out
        # ============================================================
        H_subsets = np.zeros(n_vars)
        for i in range(n_vars):
            # Get all indices except i
            idx = [j for j in range(n_vars) if j != i]
            Sigma_minus_i = Sigma[np.ix_(idx, idx)]
            
            sign, logdet_minus_i = np.linalg.slogdet(Sigma_minus_i)
            if sign <= 0:
                warnings.warn(f"Subset covariance not positive definite for X₋{i}")
                logdet_minus_i = np.log(np.abs(np.linalg.det(Sigma_minus_i)) + 1e-10)
            
            # h(X₋ᵢ) = ((n-1)/2)*log(2πe) + (1/2)*log|Σ₋ᵢ|
            const_term_subset = 0.5 * (n_vars - 1) * np.log(2 * np.pi * np.e)
            H_subsets[i] = 0.5 * logdet_minus_i + const_term_subset
        
        # ============================================================
        # Information measures
        # ============================================================
        
        # Total Correlation: TC = Σᵢ h(Xᵢ) - h(X)
        tc = H_marginals.sum() - H_joint
        
        # Dual Total Correlation: DTC = Σᵢ h(X₋ᵢ) - (n-1)*h(X)
        dtc = H_subsets.sum() - (n_vars - 1) * H_joint
        
        # O-information: Ω = TC - DTC
        omega = tc - dtc
        
        return {
            'omega': omega,
            'tc': tc,
            'dtc': dtc,
            's_inf': tc + dtc,  # Total information
            'synergy': max(0, -omega),      # Negative omega = synergy
            'redundancy': max(0, omega),    # Positive omega = redundancy
            # Additional diagnostic info
            'H_joint': H_joint,
            'H_marginals_sum': H_marginals.sum(),
            'H_subsets_sum': H_subsets.sum(),
            'det_Sigma': np.linalg.det(Sigma)
        }
    
    def _knn_oi(self, X: np.ndarray, k: int = 3) -> Dict[str, float]:
        """
        KNN-based entropy estimation (Kraskov method).
        Slower but doesn't assume Gaussian.
        
        FIXED: Same DTC fix as Gaussian method
        """
        n_samples, n_vars = X.shape
        
        # Standardize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        
        def knn_entropy(data, k=k):
            """Estimate entropy using k-nearest neighbors."""
            nbrs = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(data)
            distances, _ = nbrs.kneighbors(data)
            eps = distances[:, k]
            n, d = data.shape
            return np.log(n - 1) - np.log(k) + np.log(2) * d + d * np.mean(np.log(eps + 1e-10))
        
        # Joint entropy
        H_joint = knn_entropy(X)
        
        # Marginal entropies
        H_marginals = np.array([knn_entropy(X[:, i:i+1]) for i in range(n_vars)])
        
        # Subset entropies and conditional entropies
        H_conditionals = np.zeros(n_vars)
        H_subsets = np.zeros(n_vars)  # FIXED: Store H(X_{-i})
        for i in range(n_vars):
            idx = [j for j in range(n_vars) if j != i]
            X_minus_i = X[:, idx]
            H_minus_i = knn_entropy(X_minus_i)
            H_subsets[i] = H_minus_i  # FIXED: Save H(X_{-i})!
            H_conditionals[i] = H_joint - H_minus_i
        
        # Total Correlation
        tc = H_marginals.sum() - H_joint
        
        # Dual Total Correlation (FIXED: use H_subsets instead of H_conditionals)
        dtc = H_subsets.sum() - (n_vars - 1) * H_joint
        
        # O-information
        omega = tc - dtc
        
        return {
            'omega': omega,
            'tc': tc,
            'dtc': dtc,
            's_inf': tc + dtc,
            'synergy': max(0, -omega),      # FIXED: negative omega is synergy
            'redundancy': max(0, omega)     # FIXED: positive omega is redundancy
        }


# ============================================================================
# GRADIENT-BASED OPTIMIZATION
# ============================================================================

class OInformationGradient:
    """
    Compute O-information gradients for assessing variable contributions.
    
    Based on Scagliarini et al. (2023): "Gradients of O-information: 
    low-order descriptors of high-order dependencies"
    """
    
    @staticmethod
    def compute_gradient(
        base_genes: List[str],
        candidate_gene: str,
        data: np.ndarray,
        gene_to_idx: Dict[str, int],
        oi_estimator
    ) -> float:
        """
        Compute first-order gradient: contribution of candidate_gene to base_genes.
        
        ∂Ω = Ω(base ∪ candidate) - Ω(base)
        
        Returns:
            Gradient value (contribution to higher-order interactions)
        """
        # Base O-information
        base_indices = [gene_to_idx[g] for g in base_genes]
        X_base = data[:, base_indices]
        oi_base = oi_estimator.compute_oi(X_base)
        
        # Extended O-information
        extended_genes = base_genes + [candidate_gene]
        extended_indices = [gene_to_idx[g] for g in extended_genes]
        X_extended = data[:, extended_indices]
        oi_extended = oi_estimator.compute_oi(X_extended)
        
        gradient = oi_extended['omega'] - oi_base['omega']
        return gradient
    
    @staticmethod
    def compute_multi_gradient(
        base_genes: List[str],
        candidate_genes: List[str],
        data: np.ndarray,
        gene_to_idx: Dict[str, int],
        oi_estimator
    ) -> float:
        """
        Compute gradient for multiple candidates at once.
        
        ∂Ω = Ω(base ∪ candidates) - Ω(base)
        """
        base_indices = [gene_to_idx[g] for g in base_genes]
        X_base = data[:, base_indices]
        oi_base = oi_estimator.compute_oi(X_base)
        
        extended_genes = base_genes + candidate_genes
        extended_indices = [gene_to_idx[g] for g in extended_genes]
        X_extended = data[:, extended_indices]
        oi_extended = oi_estimator.compute_oi(X_extended)
        
        gradient = oi_extended['omega'] - oi_base['omega']
        return gradient


class GradientParticleSwarmOptimizer:
    """
    PSO that optimizes O-information gradient (not absolute omega).
    
    Finds genes that maximize contribution to base set's higher-order effects.
    """
    
    def __init__(
        self,
        n_particles: int = 20,
        n_iterations: int = 50,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        objective: str = 'max_abs',
        verbose: bool = False
    ):
        """
        Args:
            objective: 'max_abs' (maximize |gradient|),
                      'max_positive' (maximize gradient),
                      'max_negative' (minimize gradient)
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.objective = objective
        self.verbose = verbose
    
    def _evaluate_gradient(
        self,
        base_genes: List[str],
        candidate_genes: List[str],
        data: np.ndarray,
        gene_to_idx: Dict[str, int],
        oi_estimator
    ) -> float:
        """Evaluate objective function on gradient."""
        gradient = OInformationGradient.compute_multi_gradient(
            base_genes, candidate_genes, data, gene_to_idx, oi_estimator
        )
        
        if self.objective == 'max_abs':
            return abs(gradient)
        elif self.objective == 'max_positive':
            return gradient
        elif self.objective == 'max_negative':
            return -gradient
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
    
    def optimize(
        self,
        base_genes: List[str],
        candidate_genes: List[str],
        target_size: int,
        data: np.ndarray,
        gene_to_idx: Dict[str, int],
        oi_estimator
    ) -> Tuple[List[str], float, float]:
        """
        Find genes that maximize gradient contribution using PSO.
        
        Returns:
            (best_genes, final_omega, best_gradient)
        """
        n_to_add = target_size - len(base_genes)
        n_candidates = len(candidate_genes)
        
        # Initialize particles
        particles = []
        velocities = []
        for _ in range(self.n_particles):
            particle = np.zeros(n_candidates, dtype=bool)
            selected_idx = np.random.choice(n_candidates, n_to_add, replace=False)
            particle[selected_idx] = True
            particles.append(particle)
            velocities.append(np.random.randn(n_candidates) * 0.1)
        
        # Evaluate initial particles
        personal_best = particles.copy()
        personal_best_scores = []
        for particle in particles:
            selected = [candidate_genes[i] for i, sel in enumerate(particle) if sel]
            score = self._evaluate_gradient(
                base_genes, selected, data, gene_to_idx, oi_estimator
            )
            personal_best_scores.append(score)
        
        # Global best
        global_best_idx = np.argmax(personal_best_scores)
        global_best = personal_best[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        # PSO iterations
        iterator = tqdm(range(self.n_iterations), desc="PSO") if self.verbose else range(self.n_iterations)
        
        for iteration in iterator:
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (
                    self.w * velocities[i] +
                    self.c1 * r1 * (personal_best[i].astype(float) - particles[i].astype(float)) +
                    self.c2 * r2 * (global_best.astype(float) - particles[i].astype(float))
                )
                
                # Update position with sigmoid
                sigmoid = 1 / (1 + np.exp(-velocities[i]))
                new_particle = np.random.rand(n_candidates) < sigmoid
                
                # Ensure exactly n_to_add genes selected
                n_selected = new_particle.sum()
                if n_selected != n_to_add:
                    if n_selected < n_to_add:
                        available = np.where(~new_particle)[0]
                        to_add = np.random.choice(available, n_to_add - n_selected, replace=False)
                        new_particle[to_add] = True
                    else:
                        selected = np.where(new_particle)[0]
                        to_remove = np.random.choice(selected, n_selected - n_to_add, replace=False)
                        new_particle[to_remove] = False
                
                particles[i] = new_particle
                
                # Evaluate
                selected = [candidate_genes[j] for j, sel in enumerate(new_particle) if sel]
                score = self._evaluate_gradient(
                    base_genes, selected, data, gene_to_idx, oi_estimator
                )
                
                # Update personal best
                if score > personal_best_scores[i]:
                    personal_best[i] = new_particle.copy()
                    personal_best_scores[i] = score
                    
                    if score > global_best_score:
                        global_best = new_particle.copy()
                        global_best_score = score
        
        # Extract best genes
        best_selected = [candidate_genes[i] for i, sel in enumerate(global_best) if sel]
        best_genes = base_genes + best_selected
        
        # Compute final omega and gradient
        best_indices = [gene_to_idx[g] for g in best_genes]
        X_best = data[:, best_indices]
        final_oi = oi_estimator.compute_oi(X_best)
        
        base_indices = [gene_to_idx[g] for g in base_genes]
        X_base = data[:, base_indices]
        base_oi = oi_estimator.compute_oi(X_base)
        best_gradient = final_oi['omega'] - base_oi['omega']
        
        return best_genes, final_oi['omega'], best_gradient


class GradientSimulatedAnnealing:
    """
    Simulated Annealing that optimizes O-information gradient.
    """
    
    def __init__(
        self,
        n_iterations: int = 500,
        temp_init: float = 1.0,
        temp_min: float = 0.01,
        alpha: float = 0.95,
        objective: str = 'max_abs',
        verbose: bool = False
    ):
        """
        Args:
            objective: 'max_abs', 'max_positive', or 'max_negative'
        """
        self.n_iterations = n_iterations
        self.temp_init = temp_init
        self.temp_min = temp_min
        self.alpha = alpha
        self.objective = objective
        self.verbose = verbose
    
    def _evaluate_gradient(
        self,
        base_genes: List[str],
        candidate_genes: List[str],
        data: np.ndarray,
        gene_to_idx: Dict[str, int],
        oi_estimator
    ) -> float:
        """Evaluate objective function on gradient."""
        gradient = OInformationGradient.compute_multi_gradient(
            base_genes, candidate_genes, data, gene_to_idx, oi_estimator
        )
        
        if self.objective == 'max_abs':
            return abs(gradient)
        elif self.objective == 'max_positive':
            return gradient
        elif self.objective == 'max_negative':
            return -gradient
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
    
    def optimize(
        self,
        base_genes: List[str],
        candidate_genes: List[str],
        target_size: int,
        data: np.ndarray,
        gene_to_idx: Dict[str, int],
        oi_estimator
    ) -> Tuple[List[str], float, float]:
        """
        Find genes that maximize gradient contribution using SA.
        
        Returns:
            (best_genes, final_omega, best_gradient)
        """
        n_to_add = target_size - len(base_genes)
        
        # Initialize
        current_selection = set(np.random.choice(candidate_genes, n_to_add, replace=False))
        current_score = self._evaluate_gradient(
            base_genes, list(current_selection), data, gene_to_idx, oi_estimator
        )
        
        best_selection = current_selection.copy()
        best_score = current_score
        
        temp = self.temp_init
        
        iterator = tqdm(range(self.n_iterations), desc="SA") if self.verbose else range(self.n_iterations)
        
        for iteration in iterator:
            # Generate neighbor
            neighbor_selection = current_selection.copy()
            to_remove = np.random.choice(list(neighbor_selection))
            neighbor_selection.remove(to_remove)
            
            available = [g for g in candidate_genes 
                        if g not in neighbor_selection and g not in base_genes]
            if available:
                to_add = np.random.choice(available)
                neighbor_selection.add(to_add)
            else:
                neighbor_selection.add(to_remove)
            
            # Evaluate
            neighbor_score = self._evaluate_gradient(
                base_genes, list(neighbor_selection), data, gene_to_idx, oi_estimator
            )
            
            # Accept or reject
            delta = neighbor_score - current_score
            if delta > 0 or np.random.rand() < np.exp(delta / temp):
                current_selection = neighbor_selection
                current_score = neighbor_score
                
                if current_score > best_score:
                    best_selection = current_selection.copy()
                    best_score = current_score
            
            # Cool down
            temp = max(self.temp_min, temp * self.alpha)
        
        # Extract best genes
        best_genes = base_genes + list(best_selection)
        
        # Compute final omega and gradient
        best_indices = [gene_to_idx[g] for g in best_genes]
        X_best = data[:, best_indices]
        final_oi = oi_estimator.compute_oi(X_best)
        
        base_indices = [gene_to_idx[g] for g in base_genes]
        X_base = data[:, base_indices]
        base_oi = oi_estimator.compute_oi(X_base)
        best_gradient = final_oi['omega'] - base_oi['omega']
        
        return best_genes, final_oi['omega'], best_gradient


# ============================================================================
# MAIN FILTER CLASS
# ============================================================================

class FastHierarchicalOIFilter(tf_base.BaseTFIdentityPipeline):
    """
    Fast hierarchical O-information filter with gradient-based higher-order search.
    
    All-in-one implementation:
    - Fast O-information computation (Gaussian/KNN)
    - Fixed DTC calculation and sign convention
    - Gradient-based optimization (PSO/SA)
    - Exhaustive triplet search + hierarchical extension
    
    Strategy:
    1. Select top-k genes by JSD specificity
    2. Compute O-information for all triplets (fast!)
    3. Use gradient-based optimization to extend to higher orders
    4. Extract key TFs based on contribution to synergy/redundancy
    """
    
    def __init__(
        self,
        adata,
        tf_list: List[str],
        target_cell_type: str,
        cell_type_key: str = 'cell_type',
        scgx_sig_file: Optional[str] = None,
        chipseq_file: Optional[str] = None,
        known_identity_tfs: Optional[List[str]] = None,
        verbose: bool = True,
        # O-information parameters
        oi_method: str = 'gaussian',  # 'gaussian' or 'knn'
        # Higher-order search
        search_higher_orders: bool = True,
        max_order: int = 5,
        optimizer: str = 'sa',  # 'pso' or 'sa'
        gradient_objective: str = 'max_negative',  # 'max_abs', 'max_positive', 'max_negative'
        n_top_triplets: int = 5,
        # Optimizer parameters
        pso_particles: int = 20,
        pso_iterations: int = 50,
        sa_iterations: int = 1000,
        **kwargs
    ):
        """
        Initialize filter.
        
        Args:
            gradient_objective: What to optimize in gradient search:
                - 'max_abs': Maximize |∂Ω| (strongest contribution, any direction)
                - 'max_positive': Maximize +∂Ω (enhance synergy/redundancy)
                - 'max_negative': Maximize -∂Ω (oppose synergy/redundancy)
        """
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
        

        self.oi_method = oi_method
        self.search_higher_orders = search_higher_orders
        self.max_order = max_order
        self.optimizer_type = optimizer
        self.gradient_objective = gradient_objective
        self.n_top_triplets = n_top_triplets
        
        # Initialize O-information estimator
        self.oi_estimator = GaussianOInformation(method=oi_method)
        
        # Initialize gradient-based optimizer
        if optimizer == 'pso':
            self.optimizer = GradientParticleSwarmOptimizer(
                n_particles=pso_particles,
                n_iterations=pso_iterations,
                objective=gradient_objective,
                verbose=verbose
            )
        elif optimizer == 'sa':
            self.optimizer = GradientSimulatedAnnealing(
                n_iterations=sa_iterations,
                objective=gradient_objective,
                verbose=verbose
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Results storage
        self.triplet_results = None
        self.higher_order_results = {}
        
        if self.verbose:
            print(f"  O-information method: {oi_method}")
            print(f"  Optimizer: {optimizer.upper()}")
            print(f"  Gradient objective: {gradient_objective}")
    
    def apply_additional_filters(self, core_tfs: List[str]) -> List[str]:
        """
        Apply O-information filtering.
        
        Args:
            core_tfs: TFs passing core filters (high + unique expression)
            
        Returns:
            Filtered TF list
        """
        if self.verbose:
            print(f"\n[O-Information Analysis]")
            print(f"  Input TFs: {len(core_tfs)}")
        
        # Step 1: Select top genes by JSD
        top_jsd_genes = self._select_top_jsd_genes()
        
        # Step 2: Compute triplet O-information
        self.triplet_results = self._compute_triplets(top_jsd_genes)
        
        if self.verbose:
            print(f"\n  Triplet results:")
            print(f"    Total triplets: {len(self.triplet_results)}")
            if len(self.triplet_results) > 0:
                print(f"    Omega range: [{self.triplet_results['omega'].min():.4f}, {self.triplet_results['omega'].max():.4f}]")
                n_synergy = (self.triplet_results['omega'] < 0).sum()
                n_redundancy = (self.triplet_results['omega'] > 0).sum()
                print(f"    Synergistic (Ω<0): {n_synergy}")
                print(f"    Redundant (Ω>0): {n_redundancy}")
        
        # Step 3: Higher-order search (optional)
        if self.search_higher_orders and self.max_order > 3:
            remaining_tfs = [tf for tf in core_tfs if tf not in top_jsd_genes]
            
            if len(remaining_tfs) > 0:
                self._search_higher_orders(top_jsd_genes, remaining_tfs)
        
        # Step 4: Return all core TFs (filtering done at network/identity stage)
        return core_tfs
    
    def _select_top_jsd_genes(self) -> List[str]:
        """Select top genes by JSD specificity."""
        if self.results['core_filtered_tfs'] is None:
            self.apply_core_filters()
        return self.results['core_filtered_tfs']
    
    def _compute_triplets(self, gene_list: List[str]) -> pd.DataFrame:
        """Compute O-information for all triplets."""
        if self.verbose:
            print(f"\n  Computing O-information for triplets...")
        
        # Extract data
        X = self.adata_target[:, gene_list].X
        if issparse(X):
            X = X.toarray()
        
        # Standardize
        # X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        
        # Generate all triplets
        all_triplets = list(combinations(gene_list, 3))
        
        if self.verbose:
            print(f"    Total triplets: {len(all_triplets)}")
        
        results = []
        iterator = tqdm(all_triplets, desc="  Triplets") if self.verbose else all_triplets
        
        for triplet in iterator:
            indices = [gene_list.index(g) for g in triplet]
            X_triplet = X[:, indices]
            
            try:
                oi_result = self.oi_estimator.compute_oi(X_triplet)
                results.append({
                'genes': triplet,
                'omega': oi_result['omega'],
                'tc': oi_result['tc'],
                'dtc': oi_result['dtc'],
                's_inf': oi_result['s_inf'],
                'synergy': oi_result['synergy'],
                'redundancy': oi_result['redundancy'],
                'H_joint': oi_result['H_joint'],
                'H_marginals_sum': oi_result['H_marginals_sum'],
                'H_subsets_sum': oi_result['H_subsets_sum'],
                'det_Sigma': oi_result['det_Sigma']
                })
            except Exception as e:
                if self.verbose:
                    print(f"    Warning: Failed for {triplet}: {e}")
                continue
        
        df = pd.DataFrame(results)
        df = df.sort_values('omega', ascending=True)
        
        return df
    
    def _search_higher_orders(self, top_jsd_genes: List[str], remaining_tfs: List[str]):
        """Search for higher-order synergies using GRADIENT-based optimization."""
        if self.verbose:
            print(f"\n  Searching for higher-order synergies (n=4 to {self.max_order})...")
            print(f"  Using gradient objective: {self.gradient_objective}")
        
        # Prepare data
        all_genes = top_jsd_genes + remaining_tfs
        X_full = self.adata_target[:, all_genes].X
        if issparse(X_full):
            X_full = X_full.toarray()
        # X_full = (X_full - X_full.mean(axis=0)) / (X_full.std(axis=0) + 1e-10)
        
        gene_to_idx = {g: i for i, g in enumerate(all_genes)}
        
        # Get top triplets to extend
        top_triplets = self.triplet_results.head(self.n_top_triplets)
        
        if self.verbose:
            print(f"  Extending top {len(top_triplets)} triplets:")
            for idx, row in top_triplets.iterrows():
                print(f"    {row['genes']}: Ω = {row['omega']:.4f}")
        
        for order in range(4, self.max_order + 1):
            if self.verbose:
                print(f"\n  Order {order}:")
            
            order_results = []
            
            for idx, row in top_triplets.iterrows():
                base_genes = list(row['genes'])
                base_omega = row['omega']
                
                if self.verbose:
                    print(f"\n    Base: {base_genes} (Ω = {base_omega:.4f})")
                
                # GRADIENT-BASED SEARCH
                best_genes, final_omega, gradient = self.optimizer.optimize(
                    base_genes=base_genes,
                    candidate_genes=remaining_tfs,
                    target_size=order,
                    data=X_full,
                    gene_to_idx=gene_to_idx,
                    oi_estimator=self.oi_estimator
                )
                
                # Extract added genes
                added_genes = [g for g in best_genes if g not in base_genes]
                
                order_results.append({
                    'base_triplet': tuple(base_genes),
                    'base_omega': base_omega,
                    'genes': tuple(best_genes),
                    'omega': final_omega,
                    'gradient': gradient,
                    'added_genes': tuple(added_genes),
                    'order': order
                })
                
                if self.verbose:
                    print(f"    → Best extension: {added_genes}")
                    print(f"    → Final Ω = {final_omega:.4f}")
                    print(f"    → Gradient ∂Ω = {gradient:+.4f}")
                    
                    if abs(gradient) > 0.01:
                        if gradient > 0:
                            print(f"    → Enhances higher-order effects")
                        else:
                            print(f"    → Opposes higher-order effects")
            
            # Store results
            df = pd.DataFrame(order_results)
            df['abs_gradient'] = df['gradient'].abs()
            df = df.sort_values('abs_gradient', ascending=False)
            self.higher_order_results[order] = df
            
            if self.verbose:
                print(f"\n  Order {order} summary:")
                print(f"    Best gradient: {df.iloc[0]['gradient']:.4f}")
                print(f"    Best genes: {df.iloc[0]['genes']}")
    
    def identify_key_tfs(self, graph: nx.Graph, filtered_tfs: List[str]) -> List[str]:
        """Extract key identity TFs from O-information results."""
        return self._extract_key_tfs()
    
    def _extract_key_tfs(self) -> List[str]:
            """Extract final TF list from O-information results."""
            gene_counts = {}
            
            # Count from triplets (weighted by synergistic omega only)
            for _, row in self.triplet_results.head(50).iterrows():
                weight = -row['omega'] if row['omega'] < 0 else 0
                for gene in row['genes']:
                    gene_counts[gene] = gene_counts.get(gene, 0) + weight
            
            # Add from higher orders (weighted by synergistic gradient only)
            for order, df in self.higher_order_results.items():
                for _, row in df.head(50).iterrows():
                    weight = -row['gradient'] * (order / 3.0) if row['gradient'] < 0 else 0
                    for gene in row['genes']:
                        gene_counts[gene] = gene_counts.get(gene, 0) + weight
            
            # Sort and select
            sorted_tfs = sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)
            n_select = min(20, len(sorted_tfs))
            key_tfs = [tf for tf, _ in sorted_tfs[:n_select]]
            self.results['oi_weighted_counts'] = sorted_tfs
            if self.verbose:
                print(f"\n  Selected {len(key_tfs)} key TFs by O-information")
            
            return key_tfs
    
    def build_network(self, filtered_tfs: List[str]) -> nx.DiGraph:
        """Build network from O-information results."""
        G = nx.DiGraph()
        G.add_nodes_from(filtered_tfs)
        
        # Add edges from synergistic triplets
        if self.triplet_results is not None:
            synergistic = self.triplet_results[self.triplet_results['omega'] < 0]
            
            for _, row in synergistic.iterrows():
                genes = row['genes']
                omega = abs(row['omega'])
                
                if all(g in filtered_tfs for g in genes):
                    # Add pairwise edges (bidirectional for synergy)
                    for i in range(3):
                        for j in range(i+1, 3):
                            g1, g2 = genes[i], genes[j]
                            for g_from, g_to in [(g1, g2), (g2, g1)]:
                                if G.has_edge(g_from, g_to):
                                    G[g_from][g_to]['weight'] = max(G[g_from][g_to]['weight'], omega)
                                else:
                                    G.add_edge(g_from, g_to, weight=omega)
        
        return G
    
    def get_summary(self) -> Dict:
        """Get analysis summary."""
        summary = {
            'method': self.oi_method,
            'optimizer': self.optimizer_type,
            'gradient_objective': self.gradient_objective,
            'top_jsd_n': self.top_n_jsd,
            'n_triplets': len(self.triplet_results) if self.triplet_results is not None else 0
        }
        
        if self.triplet_results is not None and len(self.triplet_results) > 0:
            top = self.triplet_results.iloc[0]
            summary['top_triplet'] = {
                'genes': top['genes'],
                'omega': top['omega'],
                'tc': top['tc'],
                'dtc': top['dtc']
            }
            
            summary['triplet_stats'] = {
                'n_synergistic': (self.triplet_results['omega'] < 0).sum(),
                'n_redundant': (self.triplet_results['omega'] > 0).sum(),
                'omega_min': self.triplet_results['omega'].min(),
                'omega_max': self.triplet_results['omega'].max()
            }
        
        if self.higher_order_results:
            summary['higher_orders'] = {}
            for order, df in self.higher_order_results.items():
                if len(df) > 0:
                    top = df.iloc[0]
                    summary['higher_orders'][order] = {
                        'genes': top['genes'],
                        'omega': top['omega'],
                        'gradient': top['gradient'],
                        'base_omega': top['base_omega']
                    }
        
        return summary
    
    def filter(self) -> List[str]:
        """Run complete pipeline and return identity TFs."""
        results = self.run()
        return results['identity_tfs']