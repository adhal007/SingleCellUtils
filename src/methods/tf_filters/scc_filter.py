# specific modules
import src.methods.tf_filters.base_filter as tf_base
# base packages
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SCCPipeline(tf_base.BaseTFIdentityPipeline):
    """
    Strongly Connected Component (SCC) based pipeline for identity TF discovery.
    
    Pipeline steps:
    1. Core filters: High expression + Unique expression (from base class)
    2. Additional filter: Expression density threshold
    3. Compute significant TF-TF correlations
    4. Build network from ChIP-seq PKN
    5. Add correlated TFs connected to filtered TFs
    6. Identify largest strongly connected component
    """
    
    def __init__(
        self,
        adata,
        tf_list,
        target_cell_type,
        cell_type_key='cell_type',
        scgx_sig_file=None,
        chipseq_file=None,
        known_identity_tfs=None,
        use_collectri=True,
        corr_threshold=0.0,
        corr_qvalue=0.05,
        verbose=True
    ):
        """
        Initialize SCC pipeline.
        
        Additional Args:
            use_collectri: Whether to include CollecTRI database
            corr_threshold: Minimum correlation coefficient
            corr_qvalue: Q-value threshold for correlation significance
        """
        super().__init__(
            adata=adata,
            tf_list=tf_list,
            target_cell_type=target_cell_type,
            cell_type_key=cell_type_key,
            scgx_sig_file=scgx_sig_file,
            chipseq_file=chipseq_file,
            known_identity_tfs=known_identity_tfs,
            verbose=verbose
        )
        
        self.use_collectri = use_collectri
        self.corr_threshold = corr_threshold
        self.corr_qvalue = corr_qvalue
        
        # Storage for intermediate results
        self.correlation_results = None
        self.pkn_network = None
    
    def apply_additional_filters(self, core_tfs: List[str]) -> List[str]:
        """
        Apply expression density filter on top of core filters.
        
        Args:
            core_tfs: TFs passing high expression and uniqueness filters
            
        Returns:
            TFs also passing expression density filter
        """
        # Get expression matrix for core TFs
        X, tfs_present = self.get_expression_matrix(core_tfs)
        
        # Calculate average expression density
        p = np.sum(X > 0) / X.size
        
        # Calculate density per TF
        gene_density = np.sum(X > 0, axis=0) / X.shape[0]
        
        # Filter by density
        passed_filter = gene_density >= p
        filtered_tfs = [tf for tf, passed in zip(tfs_present, passed_filter) if passed]
        
        if self.verbose:
            print(f"  Expression density filter: {len(filtered_tfs)}/{len(tfs_present)}")
            print(f"    Average density threshold: {p:.4f}")
        
        # Compute correlations for later use
        self._compute_tf_correlations(filtered_tfs)
        
        return filtered_tfs
    
    def build_network(self, filtered_tfs: List[str]) -> nx.DiGraph:
        """
        Build network combining ChIP-seq PKN and correlation-based edges.
        
        Args:
            filtered_tfs: Filtered TF list
            
        Returns:
            NetworkX directed graph
        """
        # Load PKN
        pkn = self._load_pkn()
        
        # Filter PKN to TF-TF interactions
        pkn_tf = self._filter_pkn_to_tfs(pkn, filtered_tfs)
        
        # Get high expression TFs
        high_tfs = self.results.get('high_exp_tfs', filtered_tfs)
        
        # Build initial network from high TFs
        G = self._build_high_tf_network(pkn_tf, high_tfs)
        
        # Add correlated TFs if connected to high TFs
        if self.correlation_results is not None:
            G = self._add_correlated_tfs(G, pkn_tf, high_tfs)
        
        if self.verbose:
            print(f"  Final network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def identify_key_tfs(self, graph: nx.DiGraph, filtered_tfs: List[str]) -> List[str]:
        """
        Identify TFs in the largest strongly connected component.
        
        Args:
            graph: Gene regulatory network
            filtered_tfs: List of filtered TFs
            
        Returns:
            TFs in largest SCC
        """
        # Find all strongly connected components
        sccs = list(nx.strongly_connected_components(graph))
        
        if not sccs:
            if self.verbose:
                print("  Warning: No strongly connected components found")
            return []
        
        # Get largest SCC
        largest_scc = max(sccs, key=len)
        scc_tfs = sorted(list(largest_scc))
        
        if self.verbose:
            print(f"  Number of SCCs: {len(sccs)}")
            scc_sizes = sorted([len(scc) for scc in sccs], reverse=True)[:5]
            print(f"  Top SCC sizes: {scc_sizes}")
            print(f"  Largest SCC: {len(scc_tfs)} TFs")
        
        # Store SCC subgraph
        self.results['scc_graph'] = graph.subgraph(scc_tfs).copy()
        
        return scc_tfs
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _compute_tf_correlations(self, tfs: List[str]):
        """
        Compute pairwise TF correlations with significance testing.
        
        Args:
            tfs: List of TF names
        """
        # Get expression matrix
        X, tfs_present = self.get_expression_matrix(tfs)
        n_cells, n_tfs = X.shape
        
        if n_tfs < 2:
            self.correlation_results = None
            return
        
        # Compute correlation matrix
        cor_mat = np.corrcoef(X.T)
        
        # Compute t-statistics for significance
        t_mat = cor_mat * np.sqrt((n_cells - 2) / (1 - cor_mat**2 + 1e-10))
        p_mat = 2 * stats.t.sf(np.abs(t_mat), df=n_cells - 2)
        
        # Set diagonal to NaN
        np.fill_diagonal(p_mat, np.nan)
        np.fill_diagonal(cor_mat, np.nan)
        
        # Extract upper triangle
        upper_tri = np.triu_indices_from(p_mat, k=1)
        correlations = cor_mat[upper_tri]
        p_values = p_mat[upper_tri]
        
        # Remove NaN
        valid = ~np.isnan(p_values)
        if not np.any(valid):
            self.correlation_results = None
            return
        
        tf1_idx = upper_tri[0][valid]
        tf2_idx = upper_tri[1][valid]
        correlations = correlations[valid]
        p_values = p_values[valid]
        
        # Multiple testing correction
        _, q_values, _, _ = multipletests(p_values, method='bonferroni')
        
        # Create results DataFrame
        df_corr = pd.DataFrame({
            'TF1': [tfs_present[i] for i in tf1_idx],
            'TF2': [tfs_present[i] for i in tf2_idx],
            'correlation': correlations,
            'pval': p_values,
            'qval': q_values
        })
        
        # Filter significant
        df_sig = df_corr[
            (df_corr['qval'] < self.corr_qvalue) & 
            (df_corr['correlation'] > self.corr_threshold)
        ].copy()
        
        # Get unique correlated TFs
        if len(df_sig) > 0:
            correlated_tfs = list(set(df_sig['TF1'].tolist() + df_sig['TF2'].tolist()))
        else:
            correlated_tfs = []
        
        self.correlation_results = {
            'df_all': df_corr,
            'df_significant': df_sig,
            'correlated_tfs': correlated_tfs
        }
        
        if self.verbose:
            print(f"  Correlation analysis:")
            print(f"    Total pairs: {len(df_corr)}")
            print(f"    Significant pairs: {len(df_sig)}")
            print(f"    Correlated TFs: {len(correlated_tfs)}")
    
    def _load_pkn(self) -> pd.DataFrame:
        """Load prior knowledge network from ChIP-seq and optionally CollecTRI."""
        edges = []

   
        # Load ChIP-seq
        if self.chipseq_file:
            chipseq = pd.read_csv(self.chipseq_file, sep='\t')
            chipseq = chipseq[['TF', 'Gene']].copy()
            chipseq.columns = ['source', 'target']
            edges.append(chipseq)
        
        # Load CollecTRI if requested
        if self.use_collectri:
            try:
                # This is a placeholder - replace with actual CollecTRI loading
                collectri = pd.DataFrame()  # Load your CollecTRI data here
                if not collectri.empty:
                    edges.append(collectri[['source', 'target']])
            except:
                pass
        
        if edges:
            pkn = pd.concat(edges, ignore_index=True).drop_duplicates()
        else:
            pkn = pd.DataFrame(columns=['source', 'target'])
        
        self.pkn_network = pkn
        return pkn
    
    def _filter_pkn_to_tfs(self, pkn: pd.DataFrame, tfs: List[str]) -> pd.DataFrame:
        """Filter PKN to only TF-TF interactions."""
        tf_set = set(tfs)
        pkn_tf = pkn[
            pkn['source'].isin(tf_set) & 
            pkn['target'].isin(tf_set)
        ].copy()
        
        return pkn_tf
    
    def _build_high_tf_network(
        self, 
        pkn_tf: pd.DataFrame, 
        high_tfs: List[str]
    ) -> nx.DiGraph:
        """Build network from high TF interactions."""
        high_tf_set = set(high_tfs)
        
        # Filter edges to high TF → high TF
        edges_high = pkn_tf[
            pkn_tf['source'].isin(high_tf_set) &
            pkn_tf['target'].isin(high_tf_set)
        ]
        
        # Build graph
        G = nx.DiGraph()
        for _, row in edges_high.iterrows():
            G.add_edge(row['source'], row['target'])
        
        return G
    
    def _add_correlated_tfs(
        self,
        G: nx.DiGraph,
        pkn_tf: pd.DataFrame,
        high_tfs: List[str]
    ) -> nx.DiGraph:
        """Add correlated TFs that connect to high TFs."""
        if not self.correlation_results:
            return G
        
        high_tf_set = set(high_tfs)
        correlated_tfs = self.correlation_results['correlated_tfs']
        
        # Check each correlated TF for connections to high TFs
        for tf in correlated_tfs:
            if tf in high_tf_set:
                continue  # Already in network
            
            # Find edges connecting to high TFs
            edges_from = pkn_tf[
                (pkn_tf['source'] == tf) & 
                (pkn_tf['target'].isin(high_tf_set))
            ]
            edges_to = pkn_tf[
                (pkn_tf['target'] == tf) & 
                (pkn_tf['source'].isin(high_tf_set))
            ]
            
            # Add edges if connected
            for _, row in edges_from.iterrows():
                G.add_edge(row['source'], row['target'])
            for _, row in edges_to.iterrows():
                G.add_edge(row['source'], row['target'])
        
        return G
    
    def get_correlation_summary(self) -> pd.DataFrame:
        """Get summary of correlation results."""
        if not self.correlation_results:
            return pd.DataFrame()
        
        return self.correlation_results['df_significant']
    
    def get_scc_subgraph(self) -> nx.DiGraph:
        """Get the SCC subgraph."""
        return self.results.get('scc_graph', nx.DiGraph())