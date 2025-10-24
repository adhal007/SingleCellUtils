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
        jsd_method: Literal['jsd', 'bhattacharyya', 'geometric_jsd', 'all'] = 'jsd',
        expr_method: Literal['scgx', 'expr_density'] = 'scgx',
        main_filter: Literal['high_only', 'unique_only', 'high_and_unique'] = 'high_and_unique',
        n_processes: int = None
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
        # Parallel processing settings
        self.n_processes = n_processes or min(cpu_count() - 1, 8)
        self.chunk_size = 10  # Process genes in chunks

        # Required files
        if scgx_sig_file is None:
            raise ValueError("Single-cell gene expression signature file is required")
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
    # STATIC METHOD FOR PARALLEL PROCESSING
    # ========================================================================
    
    @staticmethod
    def _compute_jsd_single_gene_parallel(gene_data: Tuple, method: str = 'jsd') -> Tuple[str, float]:
        """
        Static method to compute JSD for a single gene (for parallel processing).
        
        Args:
            gene_data: Tuple of (gene_name, target_expr, n_target, mu_other)
            method: Divergence method
            
        Returns:
            Tuple of (gene_name, jsd_score)
        """
        gene_name, target_expr, n_target, mu_other = gene_data
        
        score_sum = 0.0
        
        for i in range(n_target):
            x_cell = target_expr[i]
            
            # Create distributions
            denom = x_cell + mu_other
            
            if denom == 0:
                score_sum += 1.0
            else:
                # p = [x_cell/(x_cell + mu_other), mu_other/(x_cell + mu_other)]
                p0 = x_cell / denom
                p1 = mu_other / denom
                
                # q = [1, 0] (reference distribution)
                q0 = 1.0
                q1 = 0.0
                
                if method == 'jsd':
                    # Standard JSD with arithmetic mean
                    m0 = 0.5 * (p0 + q0)
                    m1 = 0.5 * (p1 + q1)
                    
                    # JSD = 0.5 * KL(p||m) + 0.5 * KL(q||m)
                    kl_pm = 0.0
                    if p0 > 0:
                        kl_pm += p0 * np.log2(p0 / m0)
                    if p1 > 0:
                        kl_pm += p1 * np.log2(p1 / m1)
                    
                    kl_qm = q0 * np.log2(q0 / m0)
                    score = 0.5 * (kl_pm + kl_qm)
                    
                elif method == 'bhattacharyya':
                    bc_coefficient = np.sqrt(p0 * q0) + np.sqrt(p1 * q1)
                    score = 1 - bc_coefficient
                    
                elif method == 'geometric_jsd':
                    gm0_unnorm = np.sqrt(p0 * q0)
                    gm1_unnorm = np.sqrt(p1 * q1)
                    
                    gm_sum = gm0_unnorm + gm1_unnorm
                    if gm_sum > 0:
                        m0 = gm0_unnorm / gm_sum
                        m1 = gm1_unnorm / gm_sum
                    else:
                        m0 = 0.5 * (p0 + q0)
                        m1 = 0.5 * (p1 + q1)
                    
                    kl_pm = 0.0
                    if p0 > 0 and m0 > 0:
                        kl_pm += p0 * np.log2(p0 / m0)
                    if p1 > 0 and m1 > 0:
                        kl_pm += p1 * np.log2(p1 / m1)
                    
                    kl_qm = 0.0
                    if q0 > 0 and m0 > 0:
                        kl_qm += q0 * np.log2(q0 / m0)
                    if q1 > 0 and m1 > 0:
                        kl_qm += q1 * np.log2(q1 / m1)
                    
                    score = 0.5 * (kl_pm + kl_qm)
                
                score_sum += score
        
        return gene_name, score_sum
    
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
        passed_filter = gene_density >= p + p_std
        filtered_tfs = [tf for tf, passed in zip(tf_genes, passed_filter) if passed]
        print(f"TFs passing filter: {len(filtered_tfs)}/{len(tf_genes)}")
        
        return filtered_tfs
    
    def filter_high_expression(self) -> List[str]:
        """
        Filter TFs with high expression in target cell type.
        Uses pre-computed expression signatures.
        
        Returns:
            List of TF names with high expression
        """
        if self.expr_method == 'scgx':
            tfs_high_df = pd.read_csv(self.scgx_sig_file, sep='\t')
            high_tfs = tfs_high_df[
                tfs_high_df['expr.level'].isin(['high', 'medium'])
            ]['gene'].tolist()
            
            # Filter to only include TFs from our TF list
            high_tfs = [tf for tf in high_tfs if tf in self.tf_list]
            

        else:
            high_tfs = self.filter_tfs_by_expression_density()

        if self.verbose:
            print(f"  High expression TFs: {len(high_tfs)}/{len(self.tf_list)}")

        return high_tfs
    
    def filter_unique_expression(
        self, 
        tfs: Optional[List[str]] = None,
        jsd_threshold: float = None,
        top_n: int = None,
        top_percentile: float = 75,
        use_parallel: bool = True
    ) -> List[str]:
        """
        Filter TFs with unique expression pattern using divergence measures.
        NOTE: Lower JSD values indicate higher specificity to target cell type.
        
        Args:
            tfs: List of TFs to evaluate (if None, uses all TFs)
            jsd_threshold: Absolute threshold (e.g., score < threshold for specificity)
            top_n: Select top N TFs by score (lowest scores)
            top_percentile: Select top percentile of TFs (e.g., 75 for lowest 25%)
            use_parallel: Whether to use parallel processing
            
        Returns:
            List of TF names with unique expression
        """
        method = self.jsd_method
        if tfs is None:
            tfs = self.tf_list
        
        # Filter to TFs present in data
        tfs_present = [tf for tf in tfs if tf in self.adata.var_names]
        
        if self.verbose:
            print(f"  Computing {method} divergence for {len(tfs_present)} TFs...")
        
        if use_parallel and len(tfs_present) > 20:  # Only parallelize if worth it
            scores = self.multiprocess_jsd(tfs_present, method=method)
        # else:
        #     # Sequential computation for small gene sets
        #     scores = {}
        #     for tf in tqdm(tfs_present, disable=not self.verbose, desc=f"    Computing {method}"):
        #         score = self._compute_jsd_specificity(
        #             self.adata, tf, self.target_cell_type, self.cell_type_key, method=method
        #         )
        #         scores[tf] = score
        
        # For backward compatibility, always store as 'jsd_scores' too
        self.results['jsd_scores'] = scores
        
        # Sort TFs by score - ASCENDING order (lower is better/more specific)
        sorted_tfs = sorted(scores.items(), key=lambda x: x[1], reverse=False)  # Changed to False
        
        # Apply filtering based on provided criteria
        unique_tfs = []
        
        if jsd_threshold is not None:
            # Use absolute threshold - select scores BELOW threshold
            unique_tfs = [tf for tf, score in sorted_tfs if score <= jsd_threshold]  # Changed to <=
            if self.verbose:
                print(f"  {method} < {jsd_threshold}: {len(unique_tfs)} TFs")  # Changed to 
        
        elif top_n is not None:
            # Select top N (lowest scores)
            unique_tfs = [tf for tf, score in sorted_tfs[:min(top_n, len(sorted_tfs))]]
            if self.verbose and sorted_tfs:
                max_score = sorted_tfs[min(top_n-1, len(sorted_tfs)-1)][1]  # Changed variable name
                print(f"  Top {top_n} TFs ({method} <= {max_score:.3f})")  # Changed to <=
        
        elif top_percentile is not None:
            # Select top percentile (lowest scores)
            n_select = int(len(sorted_tfs) * (100 - top_percentile) / 100)
            n_select = max(1, n_select)  # At least 1
            unique_tfs = [tf for tf, score in sorted_tfs[:n_select]]
            if self.verbose and sorted_tfs:
                max_score = sorted_tfs[min(n_select-1, len(sorted_tfs)-1)][1]  # Changed variable name
                print(f"  Top {100-top_percentile}% TFs: {len(unique_tfs)} ({method} <= {max_score:.3f})")  # Changed to <=
        
        else:
            # Default: use statistical threshold - select scores BELOW mean - 2*std
            if sorted_tfs:
                scores_array = np.array([score for _, score in sorted_tfs])
                threshold = np.mean(scores_array) - 2 * np.std(scores_array)  # Changed to minus
                unique_tfs = [tf for tf, score in sorted_tfs if score <= threshold]  # Changed to <=
                if self.verbose:
                    print(f"  {method} < mean-2std ({threshold:.3f}): {len(unique_tfs)} TFs")  # Changed to < and mean-2std
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
        if self.main_filter == 'high_and_unique':
            # Apply high expression filter
            high_exp_tfs = self.filter_high_expression()
            self.results['high_exp_tfs'] = high_exp_tfs
            
            # Apply uniqueness filter (on high expression TFs for efficiency)
            unique_exp_tfs = self.filter_unique_expression(high_exp_tfs)
            self.results['unique_exp_tfs'] = unique_exp_tfs
            
            # Core filtered = intersection
            core_filtered = list(set(high_exp_tfs) & set(unique_exp_tfs))
            self.results['core_filtered_tfs'] = core_filtered
        elif self.main_filter == 'high_only':
            high_exp_tfs = self.filter_high_expression()
            self.results['high_exp_tfs'] = high_exp_tfs
            self.results['unique_exp_tfs'] =  high_exp_tfs
            core_filtered = list(set(high_exp_tfs) & set(unique_exp_tfs))
            self.results['core_filtered_tfs'] = core_filtered
        else:
            # Apply uniqueness filter (on high expression TFs for efficiency)
            high_exp_tfs = self.tf_list
            self.results['high_exp_tfs'] = high_exp_tfs
            unique_exp_tfs = self.filter_unique_expression(high_exp_tfs)
            self.results['unique_exp_tfs'] = unique_exp_tfs
            core_filtered = list(set(high_exp_tfs) & set(unique_exp_tfs))
            self.results['core_filtered_tfs'] = core_filtered
        if self.verbose:
            print(f"  Core filtered TFs (high & unique): {len(core_filtered)}")
        
        return core_filtered
    
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

    