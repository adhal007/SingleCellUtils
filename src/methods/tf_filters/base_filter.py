from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
from scipy.sparse import issparse
import scipy.sparse as sp
from anndata import AnnData
from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional, Set, Literal
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from scipy.stats import pearsonr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BaseTFIdentityPipeline(ABC):
    """
    Base class for TF identity discovery pipelines.
    
    Core assumptions for TF identity:
    1. High expression in target cell type
    2. Unique expression pattern (specificity to target cell type)
    
    Child classes extend with additional filters (network-based, synergy, etc.)
    """
    
    def __init__(
        self,
        adata: AnnData,
        tf_list: List[str],
        target_cell_type: str,
        cell_type_key: str = 'cell_type',
        scgx_sig_file: str = None,
        chipseq_file: str = None,
        known_identity_tfs: List[str] = None,
        verbose: bool = True,
        jsd_method: Literal['jsd', 'bhattacharyya', 'geometric_jsd', 'mmd'] = 'jsd',
        expr_method: Literal['scgx', 'expr_density', 'iqr_threshold', 'wilcoxon'] = 'wilcoxon',
        main_filter: Literal['high_only', 'unique_only', 'high_and_unique'] = 'high_and_unique',
        n_processes: int = None,
        top_n_high: int = 100,
        top_jsd_pc: float = 0.05,
        top_n_jsd: int = 50
    ):
        """
        Initialize base pipeline.
        
        Args:
            adata: AnnData object with gene expression
            tf_list: List of all TF gene names
            target_cell_type: Target cell type to find identity TFs for
            cell_type_key: Column name in adata.obs for cell types
            scgx_sig_file: Path to single-cell gene expression signature file
            chipseq_file: Optional ChIP-seq network file
            known_identity_tfs: Optional list of known identity TFs for validation
            verbose: Print progress messages
        """
        # Core data
        self.adata = adata
        self.tf_list = tf_list
        self.target_cell_type = target_cell_type
        self.cell_type_key = cell_type_key
        self.verbose = verbose
        self.jsd_method = jsd_method
        self.expr_method = expr_method
        self.main_filter = main_filter
        self.top_n_high = top_n_high
        self.top_jsd_pc = top_jsd_pc
        self.top_n_jsd = top_n_jsd
        # Parallel processing settings
        self.n_processes = n_processes or min(cpu_count() - 1, 8)
        self.chunk_size = 10  # Process genes in chunks

        # # Required files
        # if scgx_sig_file is None:
        #     raise ValueError("Single-cell gene expression signature file is required")
        self.scgx_sig_file = scgx_sig_file
        
        # Optional files
        self.chipseq_file = chipseq_file
        self.known_identity_tfs = known_identity_tfs or []
        
        # Extract target cells
        self.adata_target = adata[adata.obs[cell_type_key] == target_cell_type].copy()
        
        # Pre-compute data for JSD calculations
        self._prepare_jsd_data()
        
        # Initialize results container
        self.results = {
            'high_exp_tfs': None,
            'unique_exp_tfs': None,
            'core_filtered_tfs': None,  # Intersection of high & unique
            'final_filtered_tfs': None,  # After child class filtering
            'network': None,
            'identity_tfs': None,
            'metrics': {}
        }
        
        if self.verbose:
            self._print_initialization()
    
    def _prepare_jsd_data(self):
        """Pre-compute data needed for parallel JSD calculations."""
        # Pre-compute masks and indices
        self.target_mask = self.adata.obs[self.cell_type_key] == self.target_cell_type
        self.other_mask = ~self.target_mask
        self.target_indices = np.where(self.target_mask)[0]
        self.other_indices = np.where(self.other_mask)[0]
        
        self.n_target = len(self.target_indices)
        self.n_other = len(self.other_indices)
        
        # Pre-extract and convert expression matrix for multiprocessing
        X = self.adata.X
        if issparse(X):
            if self.verbose:
                print("  Converting sparse matrix for parallel processing...")
            self.X_dense = X.toarray()
        else:
            self.X_dense = X
        
        # Pre-extract target and other expression
        self.X_target = self.X_dense[self.target_indices, :]
        self.X_other = self.X_dense[self.other_indices, :] if self.n_other > 0 else None
        
        # Pre-compute mean expression in other cells for all genes
        if self.X_other is not None:
            self.mu_other_all = np.mean(self.X_other, axis=0)
        else:
            self.mu_other_all = np.zeros(self.X_dense.shape[1])

# ========================================================================
# METHODS FOR SPECIFICITY CALCULATION
# =======================================================================
    # ========================================================================
    # STATIC METHOD FOR PARALLEL PROCESSING
    # ========================================================================
    @staticmethod
    def _compute_jsd_single_gene_parallel(gene_data: Tuple, method: str = 'jsd') -> Tuple[str, float]:
        """
        Compute per-gene specificity score across target cells.

        Parameters
        ----------
        gene_data : Tuple
            Expected formats:
              - (gene_name, target_expr, n_target, mu_other)
                backwards-compatible: mu_other is mean expression in the 'other' population.
              - (gene_name, target_expr, n_target, mu_other, n_other, total_cells)
                extended: includes the number of 'other' cells and total cell count (used by PMI).
        method : str
            One of: 'jsd' (arithmetic JSD), 'geometric_jsd', 'alpha_jsd', 'bhattacharyya',
                    'hellinger', 'pmi'
        alpha : float
            For 'alpha_jsd' or 'geometric_jsd' (geometric_jsd uses alpha=0.5).
        Returns
        -------
        (gene_name, score_sum)
          score_sum is the sum of per-target-cell scores (matching your original API).
          Note: for methods like PMI you may prefer a single summary score (we compute a per-cell
          contribution and sum for API compatibility).
        """

        # Unpack gene_data supporting both 4- and 6-tuple formats
        if len(gene_data) == 4:
            gene_name, target_expr, n_target, mu_other = gene_data
            n_other = None
            total_cells = None
        elif len(gene_data) == 6:
            gene_name, target_expr, n_target, mu_other, n_other, total_cells = gene_data
        else:
            raise ValueError("gene_data must be a 4-tuple or 6-tuple. Received length: {}".format(len(gene_data)))

        # Ensure target_expr is an array
        target_expr = np.asarray(target_expr, dtype=float)
        if len(target_expr) != n_target:
            # warning: lengths mismatch - prefer to trust provided n_target but we'll use len()
            n_target = len(target_expr)

        eps = 1e-12  # small stabilizer for logs / denominators
        score_sum = 0.0

        # Precompute some sums if needed (PMI)
        sum_target_expr = float(np.sum(target_expr))
        if n_other is not None:
            sum_other_expr_est = float(mu_other) * float(n_other)
            total_expr_est = sum_target_expr + sum_other_expr_est
        else:
            total_expr_est = None

        # iterate per target cell (keeps original behaviour)
        for i in range(n_target):
            x_cell = float(target_expr[i])

            denom = x_cell + mu_other

            # If both are zero -> identical distributions => divergence 0.
            if denom == 0.0:
                score = 0.0
                score_sum += score
                continue

            # two-bin pmf for this cell
            p0 = x_cell / denom
            p1 = mu_other / denom

            # reference q: point mass on first bin
            q0 = 1.0
            q1 = 0.0

            # Numerical safety
            p0_safe = max(p0, eps)
            p1_safe = max(p1, eps)
            q0_safe = max(q0, eps)
            q1_safe = max(q1, eps)

            if method == 'jsd':
                # arithmetic-mean JSD
                m0 = 0.5 * (p0 + q0)
                m1 = 0.5 * (p1 + q1)
                m0 = max(m0, eps)
                m1 = max(m1, eps)

                kl_pm = 0.0
                if p0 > 0.0:
                    kl_pm += p0 * np.log2(p0_safe / m0)
                if p1 > 0.0:
                    kl_pm += p1 * np.log2(p1_safe / m1)

                kl_qm = 0.0
                # q0 = 1
                kl_qm += q0 * np.log2(q0_safe / m0)

                score = 0.5 * (kl_pm + kl_qm)

            elif method in ('geometric_jsd', 'alpha_jsd'):
                # alpha controls the geometric interpolation; alpha=0.5 is geometric JSD
                if method == 'geometric_jsd':
                    alpha_used = 0.5
                else:
                    alpha: float = 0.5
                    alpha_used = float(alpha)
                    if not (0.0 < alpha_used <= 1.0):
                        raise ValueError("alpha must be in (0,1]. Got: {}".format(alpha_used))

                # geometric interpolation in log space
                log_m0 = alpha_used * np.log(p0_safe) + (1.0 - alpha_used) * np.log(q0_safe)
                log_m1 = alpha_used * np.log(p1_safe) + (1.0 - alpha_used) * np.log(q1_safe)
                m0 = np.exp(log_m0)
                m1 = np.exp(log_m1)
                m_sum = m0 + m1
                m0 /= m_sum
                m1 /= m_sum
                m0 = max(m0, eps)
                m1 = max(m1, eps)

                kl_pm = 0.0
                if p0 > eps:
                    kl_pm += p0 * np.log2(p0_safe / m0)
                if p1 > eps:
                    kl_pm += p1 * np.log2(p1_safe / m1)

                kl_qm = 0.0
                if q0 > eps:
                    kl_qm += q0 * np.log2(q0_safe / m0)
                if q1 > eps:
                    kl_qm += q1 * np.log2(q1_safe / m1)

                score = 0.5 * (kl_pm + kl_qm)

            elif method == 'bhattacharyya':
                # BC = sum(sqrt(p_i * q_i)) = sqrt(p0 * q0) + sqrt(p1 * q1)
                bc = np.sqrt(p0_safe * q0_safe) + np.sqrt(p1_safe * q1_safe)
                # Convert BC to dissimilarity in [0,1]: 1 - BC (note: not a metric but convenient)
                # bc is <= sqrt(1*1)+... but with two bins and proper normalization bc <= 1
                score = max(0.0, 1.0 - bc)

            elif method == 'hellinger':
                # Hellinger distance: H = (1/sqrt(2)) * sqrt(sum (sqrt(p)-sqrt(q))^2)
                h = np.sqrt((np.sqrt(p0_safe) - np.sqrt(q0_safe))**2 + (np.sqrt(p1_safe) - np.sqrt(q1_safe))**2)
                score = (1.0 / np.sqrt(2.0)) * h  # in [0,1]

            elif method == 'pmi':
                # PMI requires knowledge of n_other and total_cells to compute marginals.
                if (n_other is None) or (total_expr_est is None) or (total_cells is None):
                    raise ValueError("PMI requires gene_data to contain n_other and total_cells "
                                     "(use 6-tuple: gene_name, target_expr, n_target, mu_other, n_other, total_cells).")

                # Compute approximate p(tf, c), p(tf), p(c) using expression mass
                # p(tf, c) ≈ sum_target_expr / total_expr_all
                # p(tf) ≈ (sum_target_expr + sum_other_expr_est) / total_expr_all
                # p(c) ≈ n_target / total_cells
                sum_other_expr_est = mu_other * n_other
                total_expr_all = sum_target_expr + sum_other_expr_est + eps

                p_tf_c = (sum_target_expr + eps) / total_expr_all
                p_tf = (sum_target_expr + sum_other_expr_est + eps) / total_expr_all
                p_c = (n_target + eps) / (total_cells + eps)

                # PMI (can be negative). We'll use Positive PMI (PPMI) or a normalized variant.
                raw_pmi = np.log2((p_tf_c) / (p_tf * p_c + eps) + eps)
                # Convert to a bounded positive score: e.g., positive PMI normalized by a log factor.
                # We map raw_pmi in [-inf, +inf] to [0,1] by sigmoid-like transform (but keep interpretability)
                # Simple scaling: max(0, raw_pmi) / (1 + max(0, raw_pmi))
                pos_pmi = max(0.0, raw_pmi)
                score = pos_pmi / (1.0 + pos_pmi)

            else:
                raise ValueError(f"Unknown method: {method}")

            # ensure non-negative and finite
            if not np.isfinite(score) or score < 0.0:
                score = max(0.0, float(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)))

            score_sum += score

        return gene_name, float(score_sum)        

    def compute_mmd_for_gene(self, gene: str, kernel: str = 'rbf') -> float:
        """
        Compute MMD for a single gene.
        """
        if gene not in self.adata.var_names:
            return 0.0
        
        gene_idx = np.where(self.adata.var_names == gene)[0][0]
        
        # Extract gene expression
        X_target_gene = self.X_target[:, gene_idx].reshape(-1, 1)
        X_other_gene = self.X_other[:, gene_idx].reshape(-1, 1) if self.X_other is not None else np.array([]).reshape(-1, 1)
        
        n_target = len(X_target_gene)
        n_other = len(X_other_gene)
        
        if n_target == 0 or n_other == 0:
            return 0.0
        
        # Subsample if too many cells
        max_cells = 500
        if n_target > max_cells:
            idx = np.random.choice(n_target, max_cells, replace=False)
            X_target_gene = X_target_gene[idx]
            n_target = max_cells
        
        if n_other > max_cells:
            idx = np.random.choice(n_other, max_cells, replace=False)
            X_other_gene = X_other_gene[idx]
            n_other = max_cells
        
        # Compute RBF kernel MMD
        # Auto-tune gamma
        combined = np.vstack([X_target_gene, X_other_gene])
        if combined.shape[0] > 1:

            pairwise_dists = cdist(combined, combined, 'euclidean')
            pairwise_dists = pairwise_dists[pairwise_dists > 0]
            if len(pairwise_dists) > 0:
                median_dist = np.median(pairwise_dists)
                gamma = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 1.0
            else:
                gamma = 1.0
        else:
            gamma = 1.0
        
        # Kernel matrices
        K_tt = np.exp(-gamma * cdist(X_target_gene, X_target_gene, 'sqeuclidean'))
        K_oo = np.exp(-gamma * cdist(X_other_gene, X_other_gene, 'sqeuclidean'))
        K_to = np.exp(-gamma * cdist(X_target_gene, X_other_gene, 'sqeuclidean'))
        
        # MMD^2
        term1 = np.sum(K_tt) / (n_target * n_target)
        term2 = np.sum(K_oo) / (n_other * n_other)
        term3 = 2 * np.sum(K_to) / (n_target * n_other)
        
        mmd_squared = term1 + term2 - term3
        mmd = np.sqrt(max(0, mmd_squared))
        
        return mmd



    # Static method for parallel MMD computation
    @staticmethod
    def _compute_mmd_single_gene_parallel(gene_data: Tuple) -> Tuple[str, float]:
        """Static method for parallel MMD computation."""

        
        gene_name, target_expr, other_expr, n_target_orig, n_other_orig = gene_data
        
        # Reshape for kernel computation
        X_target = target_expr.reshape(-1, 1)
        X_other = other_expr.reshape(-1, 1)
        
        n_target = n_target_orig
        n_other = n_other_orig
        
        if n_target == 0 or n_other == 0:
            return gene_name, 0.0
        
        # Subsample if needed
        max_cells = 500
        if n_target > max_cells:
            idx = np.random.choice(n_target, max_cells, replace=False)
            X_target = X_target[idx]
            n_target = max_cells
        
        if n_other > max_cells:
            idx = np.random.choice(n_other, max_cells, replace=False)
            X_other = X_other[idx]
            n_other = max_cells
        
        # Compute RBF kernel MMD
        combined = np.vstack([X_target, X_other])
        if combined.shape[0] > 1:
            pairwise_dists = cdist(combined, combined, 'euclidean')
            pairwise_dists = pairwise_dists[pairwise_dists > 0]
            if len(pairwise_dists) > 0:
                median_dist = np.median(pairwise_dists)
                gamma = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 1.0
            else:
                gamma = 1.0
        else:
            gamma = 1.0
        
        # Kernel matrices
        K_tt = np.exp(-gamma * cdist(X_target, X_target, 'sqeuclidean'))
        K_oo = np.exp(-gamma * cdist(X_other, X_other, 'sqeuclidean'))
        K_to = np.exp(-gamma * cdist(X_target, X_other, 'sqeuclidean'))
        
        # MMD^2
        term1 = np.sum(K_tt) / (n_target * n_target)
        term2 = np.sum(K_oo) / (n_other * n_other)
        term3 = 2 * np.sum(K_to) / (n_target * n_other)
        
        mmd_squared = term1 + term2 - term3
        mmd = np.sqrt(max(0, mmd_squared))
        
        return gene_name, mmd




    # ========================================================================
    # MULTIPROCESSING METHOD
    # ========================================================================
    
    def multiprocess_jsd(self, genes: List[str], method: str = 'jsd') -> Dict[str, float]:
        """
        Compute JSD scores for multiple genes using multiprocessing.
        
        Args:
            genes: List of gene names
            method: Divergence method ('jsd', 'bhattacharyya', 'geometric_jsd')
            
        Returns:
            Dictionary of gene to JSD score
        """
        # Prepare data for parallel processing
        gene_data_list = []
        
        for gene in genes:
            if gene not in self.adata.var_names:
                continue
            
            gene_idx = np.where(self.adata.var_names == gene)[0][0]
            
            # Extract expression for this gene
            target_expr = self.X_target[:, gene_idx]
            mu_other = self.mu_other_all[gene_idx]
            
            gene_data_list.append((gene, target_expr, self.n_target, mu_other))
        
        if not gene_data_list:
            return {}
        
        # Create partial function with fixed method
        compute_func = partial(self._compute_jsd_single_gene_parallel, method=method)
        
        # Parallel computation
        if self.verbose:
            print(f"    Using {self.n_processes} processes for {len(gene_data_list)} genes...")
        
        with Pool(processes=self.n_processes) as pool:
            if self.verbose:
                # With progress bar
                results = list(tqdm(
                    pool.imap(compute_func, gene_data_list, chunksize=self.chunk_size),
                    total=len(gene_data_list),
                    desc=f"    Parallel {method}",
                    disable=False
                ))
            else:
                # Without progress bar
                results = pool.map(compute_func, gene_data_list, chunksize=self.chunk_size)
        
        # Convert to dictionary
        jsd_scores = dict(results)
        
        return jsd_scores
    
    # Enhanced multiprocess method for MMD
    def multiprocess_mmd(self, genes: List[str]) -> Dict[str, float]:
        """
        Compute MMD scores for multiple genes using multiprocessing.
        """
        # Prepare data for parallel processing
        gene_data_list = []
        
        for gene in genes:
            if gene not in self.adata.var_names:
                continue
            
            gene_idx = np.where(self.adata.var_names == gene)[0][0]
            
            # For MMD, we need both target and other expression
            target_expr = self.X_target[:, gene_idx]
            other_expr = self.X_other[:, gene_idx] if self.X_other is not None else np.array([])
            
            gene_data_list.append((
                gene, 
                target_expr, 
                other_expr,
                self.n_target, 
                self.n_other
            ))
        
        if not gene_data_list:
            return {}
        
        # Create partial function
        compute_func = self._compute_mmd_single_gene_parallel
        
        # Parallel computation
        if self.verbose:
            print(f"    Using {self.n_processes} processes for MMD computation...")
        
        with Pool(processes=self.n_processes) as pool:
            if self.verbose:
                results = list(tqdm(
                    pool.imap(compute_func, gene_data_list, chunksize=self.chunk_size),
                    total=len(gene_data_list),
                    desc=f"    Computing MMD",
                    disable=False
                ))
            else:
                results = pool.map(compute_func, gene_data_list, chunksize=self.chunk_size)
        
        # Convert to dictionary
        scores = dict(results)
        
        return scores
    # ========================================================================
    # CORE FILTERING METHODS (High Expression & Uniqueness)
    # ========================================================================
    def filter_tfs_by_expression_density(
        self):
        """
        Filter TFs by expression density
        
        Keep only TFs expressed > average gene density
        (R equivalent: gene_density >= p)
        
        Args:
            adata: AnnData object
            tf_list: List of TF gene symbols
            cell_type_key: Cell type column
            target_cell_type: Target cell type
        
        Returns:
            (adata_filtered, filtered_tf_list)
        """
        

        
        print(f"\nTarget cell type: {self.target_cell_type}")
        print(f"Cells: {self.adata_target.n_obs}")
        print(f"Genes: {self.adata_target.n_vars}")
        
        # Get TF expression matrix
        tf_genes = [g for g in self.tf_list if g in self.adata_target.var_names]
        
        print(f"\nTFs in dataset: {len(tf_genes)}/{len(self.tf_list)}")
        
        adata_tfs = self.adata_target[:, tf_genes].copy()
        
        # Get expression matrix
        if issparse(adata_tfs.X):
            X = adata_tfs.X.toarray()
        else:
            X = adata_tfs.X
        
        # Calculate average density (p)
        # p = sum(all values > 0) / total values
        p = np.sum(X > 0) / X.size
        p_std = np.std(X)

        
        print(f"\nAverage expression density (p): {p:.4f}")
        print(f"\nstd expression density (p): {p_std:.4f}")
        # Calculate density for each gene
        gene_density = np.sum(X > 0, axis=0) / X.shape[0]
        
        # Filter TFs
        passed_filter = gene_density >= p
        filtered_tfs = [tf for tf, passed in zip(tf_genes, passed_filter) if passed]
        print(f"TFs passing filter: {len(filtered_tfs)}/{len(tf_genes)}")
        
        return filtered_tfs

    def filter_tfs_by_iqr(self):
        """
        Filter TFs by expression density using IQR threshold
        
        Keep only TFs with expression density > Q1 - 1.5*IQR
        (removes low-expression outliers)
        
        Args:
            self: Object with adata_target, tf_list, target_cell_type attributes
            
        Returns:
            filtered_tfs: List of TF gene symbols passing the filter
        """
        
        print(f"\nTarget cell type: {self.target_cell_type}")
        print(f"Cells: {self.adata_target.n_obs}")
        print(f"Genes: {self.adata_target.n_vars}")
        
        # Get TF expression matrix
        tf_genes = [g for g in self.tf_list if g in self.adata_target.var_names]
        
        print(f"\nTFs in dataset: {len(tf_genes)}/{len(self.tf_list)}")
        
        adata_tfs = self.adata_target[:, tf_genes].copy()
        
        # Get expression matrix
        if issparse(adata_tfs.X):
            X = adata_tfs.X.toarray()
        else:
            X = adata_tfs.X
        
        # Calculate density for each gene (proportion of cells expressing it)
        gene_density = np.sum(X > 0, axis=0) / X.shape[0]
        
        # Calculate IQR statistics
        q1 = np.percentile(gene_density, 25)
        q3 = np.percentile(gene_density, 75)
        iqr = q3 - q1
        median = np.median(gene_density)
        
        # IQR outlier threshold (lower bound)
        lower_threshold = q1 - 1.5 * iqr
        
        print(f"\nExpression density statistics:")
        print(f"  Q1 (25th percentile): {q1:.4f}")
        print(f"  Median (50th percentile): {median:.4f}")
        print(f"  Q3 (75th percentile): {q3:.4f}")
        print(f"  IQR: {iqr:.4f}")
        print(f"  Lower threshold (Q1 - 1.5*IQR): {lower_threshold:.4f}")
        
        # Filter TFs - keep those above lower threshold
        passed_filter = gene_density > lower_threshold
        filtered_tfs = [tf for tf, passed in zip(tf_genes, passed_filter) if passed]
        
        print(f"\nTFs passing filter: {len(filtered_tfs)}/{len(tf_genes)}")
        print(f"TFs removed: {len(tf_genes) - len(filtered_tfs)}")
        
        return filtered_tfs
    
    def filter_by_wilcoxon(self):
        # Get TF expression matrix
        tf_genes = [g for g in self.tf_list if g in self.adata_target.var_names]

        adata_tfs = self.adata[:, tf_genes].copy()
        # Very fast implementation
        sc.tl.rank_genes_groups(adata_tfs  , groupby=self.cell_type_key, 
                                method='wilcoxon',
                                reference='rest',
                                groups=[self.target_cell_type])

        # Extract top markers
        markers = sc.get.rank_genes_groups_df(adata_tfs, group=self.target_cell_type)
        top_genes = markers[markers['pvals_adj'] < 0.05].head(self.top_n_high)
        high_tfs = top_genes['names'].to_list()
        return high_tfs
    
    def filter_high_expression(self) -> List[str]:
        """
        Filter TFs with high expression in target cell type.
        Uses pre-computed expression signatures.
        
        Returns:
            List of TF names with high expression
        """
        if self.expr_method == 'scgx':
            tfs_high_df = pd.read_csv(self.scgx_sig_file, sep='\t')
            iqr_thresh = np.percentile(tfs_high_df['not.0.perc'], 75) + 1.5*(np.percentile(tfs_high_df['not.0.perc'], 75) - np.percentile(tfs_high_df['not.0.perc'], 25)) 
            high_tfs = tfs_high_df[tfs_high_df['not.0.perc'] > iqr_thresh]['gene'].tolist()
            # high_tfs = tfs_high_df[
            #     tfs_high_df['expr.level'].isin(['high', 'medium'])
            # ]['gene'].tolist()
            
            # Filter to only include TFs from our TF list
            high_tfs = [tf for tf in high_tfs if tf in self.tf_list]
            

        elif self.expr_method == 'expr_density':
            high_tfs = self.filter_tfs_by_expression_density()

        elif self.expr_method == 'iqr_threshold':
            high_tfs = self.filter_tfs_by_iqr()
        
        elif self.expr_method == 'wilcoxon':
            high_tfs = self.filter_by_wilcoxon()
        
        if self.verbose:
            print(f"  High expression TFs: {len(high_tfs)}/{len(self.tf_list)}")

        return high_tfs
    @staticmethod
    def _find_jsd_threshold_simple(jsd_values, method='iqr'):
        """Simple, interpretable threshold based on distribution shape"""
        
        sorted_jsd = np.sort(jsd_values)
        
        if method == 'iqr':
            # Use interquartile range - classic outlier detection
            q1 = np.percentile(jsd_values, 25)
            q3 = np.percentile(jsd_values, 75)
            iqr = q3 - q1
            
            # Threshold = Q1 - 1.5*IQR (lower outliers = more specific)
            threshold = q1 - 1.5 * iqr
            
            # Ensure threshold is positive
            threshold = max(threshold, np.min(jsd_values) * 0.5)
            
            n_kept = np.sum(jsd_values <= threshold)
            
            print(f"  IQR-based threshold: {threshold:.2f}")
            print(f"    Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
            print(f"    Keeps {n_kept}/{len(jsd_values)} TFs ({100*n_kept/len(jsd_values):.1f}%)")
            
        elif method == 'mad':
            # Median Absolute Deviation - robust to outliers
            median = np.median(jsd_values)
            mad = np.median(np.abs(jsd_values - median))
            
            # For skewed distribution, use median - 2*MAD to get lower tail
            threshold = median - 2 * mad
            threshold = max(threshold, np.min(jsd_values) * 0.5)
            
            n_kept = np.sum(jsd_values <= threshold)
            
            print(f"  MAD-based threshold: {threshold:.2f}")
            print(f"    Median={median:.2f}, MAD={mad:.2f}")
            print(f"    Keeps {n_kept}/{len(jsd_values)} TFs ({100*n_kept/len(jsd_values):.1f}%)")
        
        elif method == 'top_n_percent':
            # Simply take top N% most specific
            percentile = 30  # adjust as needed
            threshold = np.percentile(jsd_values, percentile)
            n_kept = np.sum(jsd_values <= threshold)
            
            print(f"  Top {percentile}% threshold: {threshold:.2f}")
            print(f"    Keeps {n_kept}/{len(jsd_values)} TFs")
        
        return threshold

    def filter_unique_expression(
        self, 
        tfs: Optional[List[str]] = None,
        jsd_threshold: float = None,
        jsd_thresh_method: str = 'iqr',  # NEW: method for auto threshold
        top_n: int = None,
        top_percentile: float = None,
        use_parallel: bool = True
    ) -> List[str]:
        """
        Filter TFs with unique expression pattern using divergence measures.
        NOTE: Lower JSD values indicate higher specificity to target cell type.
        
        Args:
            tfs: List of TFs to evaluate (if None, uses all TFs)
            jsd_threshold: Absolute threshold (if None, auto-compute using jsd_method)
            jsd_method: Method for auto threshold ('iqr', 'mad', 'top_n_percent')
            top_n: Select top N TFs by score (lowest scores)
            top_percentile: Select top percentile of TFs (e.g., 75 for lowest 25%)
            use_parallel: Whether to use parallel processing
            
        Returns:
            List of TF names with unique expression
        """
        method = self.jsd_method
        top_percentile = self.top_jsd_pc 
        if tfs is None:
            tfs = self.tf_list
        
        # Filter to TFs present in data
        tfs_present = [tf for tf in tfs if tf in self.adata.var_names]
        
        if self.verbose:
            print(f"  Computing {method} divergence for {len(tfs_present)} TFs...")
        if method != 'mmd':
            if use_parallel and len(tfs_present) > 20:  # Only parallelize if worth it
                scores = self.multiprocess_jsd(tfs_present, method=method)
        else:
            if use_parallel and len(tfs_present) > 20:
                scores = self.multiprocess_mmd(tfs_present)

        self.results['jsd_scores'] = scores
        
        # Sort TFs by score - ASCENDING order (lower is better/more specific)
        sorted_tfs = sorted(scores.items(), key=lambda x: x[1], reverse=False)
        
        # Apply filtering based on provided criteria
        unique_tfs = []
        
        if jsd_threshold is not None:
            # Use absolute threshold - select scores BELOW threshold
            unique_tfs = [tf for tf, score in sorted_tfs if score <= jsd_threshold]
            if self.verbose:
                print(f"  {method} <= {jsd_threshold}: {len(unique_tfs)} TFs")
        
        elif top_n is not None:
            # Select top N (lowest scores)
            unique_tfs = [tf for tf, score in sorted_tfs[:min(top_n, len(sorted_tfs))]]
            if self.verbose and sorted_tfs:
                max_score = sorted_tfs[min(top_n-1, len(sorted_tfs)-1)][1]
                print(f"  Top {top_n} TFs ({method} <= {max_score:.3f})")
        
        elif top_percentile is not None:
            # Select top percentile (lowest scores)
            n_select = int(len(sorted_tfs) * (100 - top_percentile) / 100)
            n_select = max(1, n_select)  # At least 1
            unique_tfs = [tf for tf, score in sorted_tfs[:n_select]]
            if self.verbose and sorted_tfs:
                max_score = sorted_tfs[min(n_select-1, len(sorted_tfs)-1)][1]
                print(f"  Top {100-top_percentile}% TFs: {len(unique_tfs)} ({method} <= {max_score:.3f})")
        
        else:

            # # Default: use data-driven statistical threshold
            if sorted_tfs:
            #     scores_array = np.array([score for _, score in sorted_tfs])
                
            #     # Use the simple threshold finder
            #     threshold = self._find_jsd_threshold_simple(scores_array, method=jsd_thresh_method)
                all_jsd_scores = self.multiprocess_jsd(self.tf_list, method='geometric_jsd')
                all_jsd_scores_df = pd.Series(all_jsd_scores).reset_index()
                all_jsd_scores_df = all_jsd_scores_df[all_jsd_scores_df[0] >= 1]
                threshold = np.percentile(all_jsd_scores_df[0], 75) - 1.5*(np.percentile(all_jsd_scores_df[0], 75) - np.percentile(all_jsd_scores_df[0], 25))
                unique_tfs = [tf for tf, score in sorted_tfs if score <= threshold]
                if self.verbose:
                    print(f"  Data-driven threshold ({jsd_thresh_method}): {len(unique_tfs)} TFs selected")
            else:
                unique_tfs = []
        
        if self.verbose:
            print(f"  Unique expression TFs: {len(unique_tfs)}/{len(tfs_present)}")
            if sorted_tfs:
                top_5 = sorted_tfs[:min(5, len(sorted_tfs))]
                print(f"  Top 5 by {method} (most specific): {', '.join([f'{tf}({score:.2f})' for tf, score in top_5])}")
        
        return unique_tfs

    def apply_core_filters(self) -> List[str]:
        """
        Apply both core filters: high expression AND uniqueness.
        This is the foundation that all child classes build upon.
        
        Returns:
            List of TFs passing both core filters
        """

        if self.verbose:
            print("\n[Core Filtering]")
        #  Literal['high_only', 'unique_only', 'high_and_unique']
        if self.top_n_jsd is not None:
            if self.main_filter == 'high_and_unique':
                # Apply high expression filter            
                # Core filtered = intersection
                high_exp_tfs = self.filter_high_expression()
                unique_exp_tfs = self.filter_unique_expression(high_exp_tfs,top_n=self.top_n_jsd)
                self.results['high_exp_tfs'] = high_exp_tfs
                self.results['unique_exp_tfs'] = unique_exp_tfs
                self.results['core_filtered_tfs'] = unique_exp_tfs
            elif self.main_filter == 'high_only':
                high_exp_tfs = self.filter_high_expression()
                self.results['high_exp_tfs'] = high_exp_tfs
                self.results['unique_exp_tfs'] =  high_exp_tfs
                self.results['core_filtered_tfs'] = high_exp_tfs
            else:
                # Apply uniqueness filter (on high expression TFs for efficiency)
                high_exp_tfs = self.tf_list
                self.results['high_exp_tfs'] = high_exp_tfs
                unique_exp_tfs = self.filter_unique_expression(high_exp_tfs, top_n=self.top_n_jsd)
                self.results['unique_exp_tfs'] = unique_exp_tfs
                self.results['core_filtered_tfs'] = unique_exp_tfs
        elif self.top_jsd_pc is not None:
            if self.main_filter == 'high_and_unique':
                # Apply high expression filter            
                # Core filtered = intersection
                high_exp_tfs = self.filter_high_expression()
                unique_exp_tfs = self.filter_unique_expression(high_exp_tfs,top_percentile=self.top_jsd_pc)
                self.results['high_exp_tfs'] = high_exp_tfs
                self.results['unique_exp_tfs'] = unique_exp_tfs
                self.results['core_filtered_tfs'] = unique_exp_tfs
            elif self.main_filter == 'high_only':
                high_exp_tfs = self.filter_high_expression()
                self.results['high_exp_tfs'] = high_exp_tfs
                self.results['unique_exp_tfs'] =  high_exp_tfs
                self.results['core_filtered_tfs'] = high_exp_tfs
            else:
                # Apply uniqueness filter (on high expression TFs for efficiency)
                high_exp_tfs = self.tf_list
                self.results['high_exp_tfs'] = high_exp_tfs
                unique_exp_tfs = self.filter_unique_expression(high_exp_tfs, top_percentile=self.top_jsd_pc)
                self.results['unique_exp_tfs'] = unique_exp_tfs
                self.results['core_filtered_tfs'] = unique_exp_tfs
        else:
            if self.main_filter == 'high_and_unique':
                # Apply high expression filter            
                # Core filtered = intersection
                high_exp_tfs = self.filter_high_expression()
                unique_exp_tfs = self.filter_unique_expression(high_exp_tfs)
                self.results['high_exp_tfs'] = high_exp_tfs
                self.results['unique_exp_tfs'] = unique_exp_tfs
                self.results['core_filtered_tfs'] = list(
                    set(high_exp_tfs) & set(unique_exp_tfs)
                )
            elif self.main_filter == 'high_only':
                high_exp_tfs = self.filter_high_expression()
                self.results['high_exp_tfs'] = high_exp_tfs
                self.results['unique_exp_tfs'] =  high_exp_tfs
                self.results['core_filtered_tfs'] = high_exp_tfs
            else:
                # Apply uniqueness filter (on high expression TFs for efficiency)
                high_exp_tfs = self.tf_list
                self.results['high_exp_tfs'] = high_exp_tfs
                unique_exp_tfs = self.filter_unique_expression(high_exp_tfs)
                self.results['unique_exp_tfs'] = unique_exp_tfs
                self.results['core_filtered_tfs'] = unique_exp_tfs
        if self.verbose:
            print(f"Core filtered TFs (high & unique): {len(self.results['unique_exp_tfs'])}")
        
        return self.results['core_filtered_tfs'] 
    
    # ========================================================================
    # ABSTRACT METHODS FOR CHILD CLASSES
    # ========================================================================
    
    @abstractmethod
    def apply_additional_filters(self, core_tfs: List[str]) -> List[str]:
        """
        Apply strategy-specific additional filtering beyond core filters.
        
        Args:
            core_tfs: TFs that passed high expression and uniqueness filters
            
        Returns:
            List of TFs after additional filtering
        """
        pass
    
    @abstractmethod
    def build_network(self, filtered_tfs: List[str]) -> nx.DiGraph:
        """
        Build gene regulatory network from filtered TFs.
        
        Args:
            filtered_tfs: List of filtered TF names
            
        Returns:
            NetworkX directed graph
        """
        pass
    
    @abstractmethod
    def identify_key_tfs(self, graph: nx.DiGraph, filtered_tfs: List[str]) -> List[str]:
        """
        Identify key identity TFs from network using strategy-specific criteria.
        
        Args:
            graph: Gene regulatory network
            filtered_tfs: List of filtered TFs
            
        Returns:
            List of identity TF names
        """
        pass
    
    # ========================================================================
    # MAIN PIPELINE EXECUTION
    # ========================================================================
    
    def run(self) -> Dict:
        """
        Execute complete pipeline:
        1. Apply core filters (high expression & uniqueness)
        2. Apply child class additional filters
        3. Build network
        4. Identify key TFs
        5. Compute metrics
        
        Returns:
            Dictionary with all results
        """
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "="*80)
            print(f"RUNNING: {self.__class__.__name__}")
            print("="*80)
        
        # Step 1: Apply core filters
        core_tfs = self.apply_core_filters()
        
        # Step 2: Apply additional filters (child class specific)
        if self.verbose:
            print("\n[Additional Filtering]")
        final_filtered_tfs = self.apply_additional_filters(core_tfs)
        self.results['final_filtered_tfs'] = final_filtered_tfs
        
        if self.verbose:
            print(f"  Final filtered TFs: {len(final_filtered_tfs)}")
        
        # Step 3: Build network
        if self.verbose:
            print("\n[Network Construction]")
        graph = self.build_network(final_filtered_tfs)
        self.results['network'] = graph
        
        if self.verbose:
            print(f"  Network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Step 4: Identify key TFs
        if self.verbose:
            print("\n[Identity TF Selection]")
        identity_tfs = self.identify_key_tfs(graph, final_filtered_tfs)
        self.results['identity_tfs'] = identity_tfs
        
        
        if self.verbose:
            print(f"  Identity TFs found: {len(identity_tfs)}")
        
        # Step 5: Compute metrics
        elapsed = time.time() - start_time
        self.results['metrics'] = self._compute_metrics(elapsed)
        
        if self.verbose:
            self._print_summary()
        
        return self.results
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _compute_metrics(self, runtime: float) -> Dict:
        """
        Compute performance metrics.
        
        Args:
            runtime: Pipeline execution time
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'runtime': runtime,
            'n_input_tfs': len(self.tf_list),
            'n_high_exp': len(self.results['high_exp_tfs'] or []),
            'n_unique_exp': len(self.results['unique_exp_tfs'] or []),
            'n_core_filtered': len(self.results['core_filtered_tfs'] or []),
            'n_final_filtered': len(self.results['final_filtered_tfs'] or []),
            'n_identity_tfs': len(self.results['identity_tfs'] or [])
        }
        
        # Network metrics
        if self.results['network']:
            metrics['n_nodes'] = self.results['network'].number_of_nodes()
            metrics['n_edges'] = self.results['network'].number_of_edges()
        
        # Validation metrics if known TFs provided
        if self.known_identity_tfs and self.results['identity_tfs']:
            known_set = set(self.known_identity_tfs)
            found_set = set(self.results['identity_tfs'])
            captured = known_set & found_set
            
            metrics['validation'] = {
                'known_total': len(known_set),
                'captured': len(captured),
                'recall': len(captured) / len(known_set) if known_set else 0,
                'precision': len(captured) / len(found_set) if found_set else 0,
                'captured_tfs': list(captured),
                'missed_tfs': list(known_set - found_set),
                'false_positives': list(found_set - known_set)
            }
        
        return metrics
    
    def _print_initialization(self):
        """Print initialization info."""
        print("="*80)
        print(f"INITIALIZED: {self.__class__.__name__}")
        print("="*80)
        print(f"Target cell type: {self.target_cell_type}")
        print(f"Target cells: {self.adata_target.n_obs}")
        print(f"Total cells: {self.adata.n_obs}")
        print(f"Input TFs: {len(self.tf_list)}")
        if self.known_identity_tfs:
            print(f"Known identity TFs: {len(self.known_identity_tfs)}")
    
    def _print_summary(self):
        """Print pipeline results summary."""
        m = self.results['metrics']
        
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        print("\nFiltering Cascade:")
        print(f"  Input TFs:        {m['n_input_tfs']:4d}")
        print(f"  ├─ High expr:     {m['n_high_exp']:4d}")
        print(f"  ├─ Unique expr:   {m['n_unique_exp']:4d}")
        print(f"  ├─ Core (both):   {m['n_core_filtered']:4d}")
        print(f"  └─ Final:         {m['n_final_filtered']:4d}")
        
        if 'n_nodes' in m:
            print("\nNetwork:")
            print(f"  Nodes:            {m['n_nodes']:4d}")
            print(f"  Edges:            {m['n_edges']:4d}")
        
        print("\nIdentity TFs:")
        print(f"  Found:            {m['n_identity_tfs']:4d}")
        
        if 'validation' in m:
            v = m['validation']
            print("\nValidation:")
            print(f"  Recall:     {v['recall']*100:5.1f}% ({v['captured']}/{v['known_total']})")
            print(f"  Precision:  {v['precision']*100:5.1f}% ({v['captured']}/{m['n_identity_tfs']})")
            if v['captured_tfs']:
                print(f"  Captured:   {', '.join(v['captured_tfs'][:5])}{'...' if len(v['captured_tfs']) > 5 else ''}")
        
        print(f"\nRuntime: {m['runtime']:.2f}s")
    
    def load_chipseq_network(self) -> pd.DataFrame:
        """Fixed version that properly loads ChIP-seq with header."""
        if self.chipseq_file is None:
            return pd.DataFrame(columns=['source', 'target'])
        
        # Load with header
        chipseq = pd.read_csv(self.chipseq_file, sep='\t')
        
        # Use TF as source and Gene as target
        chipseq_network = chipseq[['TF', 'Gene']].copy()
        chipseq_network.columns = ['source', 'target']
        
        # Remove any duplicates
        chipseq_network = chipseq_network.drop_duplicates()
        
        return chipseq_network
    
    def get_expression_matrix(self, genes: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Get expression matrix for specified genes in target cells.
        
        Args:
            genes: List of gene names
            
        Returns:
            Expression matrix and list of genes present in data
        """
        genes_present = [g for g in genes if g in self.adata_target.var_names]
        adata_subset = self.adata_target[:, genes_present].copy()
        
        X = adata_subset.X.toarray() if issparse(adata_subset.X) else adata_subset.X
        return X, genes_present
    
    # def compute_all_divergence_metrics(
    #     self, 
    #     adata: AnnData,
    #     gene: str,
    #     target_cell_type: str,
    #     cell_type_key: str = 'cell_type'
    # ) -> dict:
    #     """
    #     Compute all divergence metrics for comparison.
        
    #     Returns:
    #         Dictionary with 'jsd', 'bhattacharyya', and 'geometric_jsd' scores
    #     """
    #     return {
    #         'jsd': self._compute_jsd_specificity(adata, gene, target_cell_type, cell_type_key, method='jsd'),
    #         'bhattacharyya': self._compute_jsd_specificity(adata, gene, target_cell_type, cell_type_key, method='bhattacharyya'),
    #         'geometric_jsd': self._compute_jsd_specificity(adata, gene, target_cell_type, cell_type_key, method='geometric_jsd')
    #     }

    