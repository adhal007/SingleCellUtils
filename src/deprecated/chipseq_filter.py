# specific modules
import src.methods.tf_filters.base_filter as tf_base

# base packages
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Optional, Tuple
from scipy.sparse import issparse
import warnings
warnings.filterwarnings('ignore')

class ChIPseqNetworkPipeline(tf_base.BaseTFIdentityPipeline):
    """
    ChIP-seq network-based pipeline for identity TF discovery.
    
    Strategy:
    1. Core filters: High expression + Unique expression (from base class)
    2. Additional filter: Network connectivity (must be in ChIP-seq network)
    3. Network: ChIP-seq derived regulatory network
    4. Identity TFs: Top centrality nodes in network
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
        jsd_threshold=0.5,
        centrality_method='pagerank',
        top_k=20,
        verbose=True
    ):
        """
        Initialize ChIP-seq network pipeline.
        
        Additional Args:
            jsd_threshold: Threshold for uniqueness filtering (0-1)
            centrality_method: Method for ranking TFs ('pagerank', 'degree', 'betweenness')
            top_k: Number of top TFs to select as identity TFs
        """
        # Validate ChIP-seq requirement
        if chipseq_file is None:
            raise ValueError("ChIP-seq network file is required for ChIPseqNetworkPipeline")
        
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
        
        # Pipeline-specific parameters
        self.jsd_threshold = jsd_threshold
        self.centrality_method = centrality_method
        self.top_k = top_k
        
        # Load ChIP-seq network once
        self.chipseq_network = self._load_and_validate_chipseq()
        
        if self.verbose:
            print(f"ChIP-seq network: {len(self.chipseq_network)} edges")
            print(f"Centrality method: {centrality_method}")
            print(f"Top-k selection: {top_k}")
    
    def _load_and_validate_chipseq(self) -> pd.DataFrame:
        """Load and validate ChIP-seq network."""
        chipseq = self.load_chipseq_network()
        
        if chipseq.empty:
            raise ValueError("ChIP-seq network is empty")
        
        # Filter to only TF-TF interactions
        tf_set = set(self.tf_list)
        chipseq = chipseq[
            chipseq['source'].isin(tf_set) & 
            chipseq['target'].isin(tf_set)
        ].copy()
        
        # Remove self-loops
        chipseq = chipseq[chipseq['source'] != chipseq['target']]
        
        return chipseq
    
    def apply_additional_filters(self, core_tfs: List[str]) -> List[str]:
        """
        Apply network-based filtering on top of core filters.
        Keep only TFs that appear in the ChIP-seq network.
        
        Args:
            core_tfs: TFs passing high expression and uniqueness filters
            
        Returns:
            TFs that are also present in ChIP-seq network
        """
        # Get all TFs present in network (as source or target)
        network_tfs = set(self.chipseq_network['source']) | set(self.chipseq_network['target'])
        
        # Filter to TFs in both core set and network
        filtered_tfs = [tf for tf in core_tfs if tf in network_tfs]
        
        if self.verbose:
            print(f"  Network-present TFs: {len(filtered_tfs)}/{len(core_tfs)}")
            dropped = set(core_tfs) - set(filtered_tfs)
            if dropped and len(dropped) <= 10:
                print(f"  Dropped (not in network): {list(dropped)}")
        
        # Store for analysis
        self.results['network_filtered_tfs'] = filtered_tfs
        
        return filtered_tfs
    
    def build_network(self, filtered_tfs: List[str]) -> nx.DiGraph:
        """
        Build directed regulatory network from ChIP-seq data.
        
        Args:
            filtered_tfs: Filtered TF list
            
        Returns:
            NetworkX directed graph
        """
        # Initialize directed graph
        G = nx.DiGraph()
        
        # Add all filtered TFs as nodes
        G.add_nodes_from(filtered_tfs)
        
        # Add node attributes
        for tf in filtered_tfs:
            # Add expression level if available
            if hasattr(self, 'results') and 'jsd_scores' in self.results:
                G.nodes[tf]['jsd_score'] = self.results['jsd_scores'].get(tf, 0)
            
            # Add known identity TF label
            G.nodes[tf]['is_known'] = tf in self.known_identity_tfs
        
        # Add edges from ChIP-seq
        tf_set = set(filtered_tfs)
        edges_added = 0
        
        for _, row in self.chipseq_network.iterrows():
            source, target = row['source'], row['target']
            
            # Only add edge if both nodes in filtered set
            if source in tf_set and target in tf_set:
                G.add_edge(source, target)
                edges_added += 1
        
        if self.verbose:
            print(f"  Added {edges_added} edges between filtered TFs")
            
            # Report basic network statistics
            if G.number_of_nodes() > 0:
                in_degrees = dict(G.in_degree())
                out_degrees = dict(G.out_degree())
                
                print(f"  Average in-degree: {np.mean(list(in_degrees.values())):.2f}")
                print(f"  Average out-degree: {np.mean(list(out_degrees.values())):.2f}")
                
                # Find isolated nodes
                isolated = [n for n in G.nodes() if G.degree(n) == 0]
                if isolated:
                    print(f"  Isolated nodes: {len(isolated)}")
        
        # Store network statistics
        self.results['network_stats'] = self._compute_network_stats(G)
        
        return G
    
    def identify_key_tfs(self, graph: nx.DiGraph, filtered_tfs: List[str]) -> List[str]:
        """
        Identify key TFs using network centrality measures.
        
        Args:
            graph: Gene regulatory network
            filtered_tfs: List of filtered TFs
            
        Returns:
            List of identity TFs based on centrality
        """
        if graph.number_of_edges() == 0:
            if self.verbose:
                print("  Warning: Network has no edges, using JSD scores for ranking")
            
            # Fallback: rank by JSD scores if available
            if 'jsd_scores' in self.results:
                scores = self.results['jsd_scores']
                ranked = sorted(
                    [(tf, scores.get(tf, 0)) for tf in filtered_tfs],
                    key=lambda x: x[1],
                    reverse=True
                )
                return [tf for tf, _ in ranked[:self.top_k]]
            else:
                # Last resort: return first top_k TFs
                return filtered_tfs[:self.top_k]
        
        # Compute centrality scores
        centrality_scores = self._compute_centrality(graph)
        
        # Rank TFs by centrality
        ranked_tfs = sorted(
            centrality_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top-k TFs
        identity_tfs = [tf for tf, score in ranked_tfs[:self.top_k]]
        
        if self.verbose:
            print(f"  Top {min(5, len(identity_tfs))} TFs by {self.centrality_method}:")
            for i, (tf, score) in enumerate(ranked_tfs[:5]):
                known_marker = " [KNOWN]" if tf in self.known_identity_tfs else ""
                print(f"    {i+1}. {tf}: {score:.4f}{known_marker}")
        
        # Store centrality scores for all TFs
        self.results['centrality_scores'] = centrality_scores
        self.results['centrality_ranking'] = ranked_tfs
        
        return identity_tfs
    
    def _compute_centrality(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        Compute centrality scores using specified method.
        
        Args:
            graph: Network graph
            
        Returns:
            Dictionary of TF to centrality score
        """
        if self.centrality_method == 'pagerank':
            # PageRank for directed networks
            try:
                scores = nx.pagerank(graph, max_iter=200)
            except nx.PowerIterationFailedConvergence:
                if self.verbose:
                    print("  PageRank failed to converge, using degree centrality")
                scores = dict(graph.degree())
                # Normalize
                max_score = max(scores.values()) if scores else 1
                scores = {k: v/max_score for k, v in scores.items()}
                
        elif self.centrality_method == 'degree':
            # Total degree (in + out)
            scores = dict(graph.degree())
            # Normalize by max degree
            max_degree = max(scores.values()) if scores else 1
            scores = {k: v/max_degree for k, v in scores.items()}
            
        elif self.centrality_method == 'in_degree':
            # In-degree (regulated by many)
            scores = dict(graph.in_degree())
            max_degree = max(scores.values()) if scores else 1
            scores = {k: v/max_degree for k, v in scores.items()}
            
        elif self.centrality_method == 'out_degree':
            # Out-degree (regulates many)
            scores = dict(graph.out_degree())
            max_degree = max(scores.values()) if scores else 1
            scores = {k: v/max_degree for k, v in scores.items()}
            
        elif self.centrality_method == 'betweenness':
            # Betweenness centrality
            scores = nx.betweenness_centrality(graph)
            
        elif self.centrality_method == 'closeness':
            # Closeness centrality
            scores = nx.closeness_centrality(graph)
            
        elif self.centrality_method == 'eigenvector':
            # Eigenvector centrality
            try:
                scores = nx.eigenvector_centrality(graph, max_iter=200)
            except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                if self.verbose:
                    print("  Eigenvector centrality failed, using PageRank")
                scores = nx.pagerank(graph)
        else:
            raise ValueError(f"Unknown centrality method: {self.centrality_method}")
        
        # Combine with JSD scores if available (weighted average)
        if 'jsd_scores' in self.results:
            jsd_scores = self.results['jsd_scores']
            combined_scores = {}
            
            for tf in scores:
                network_score = scores[tf]
                jsd_score = jsd_scores.get(tf, 0)
                # Weighted combination (70% network, 30% expression uniqueness)
                combined_scores[tf] = 0.7 * network_score + 0.3 * jsd_score
            
            return combined_scores
        
        return scores
    
    def _compute_network_stats(self, graph: nx.DiGraph) -> Dict:
        """
        Compute detailed network statistics.
        
        Args:
            graph: Network graph
            
        Returns:
            Dictionary of network statistics
        """
        stats = {
            'n_nodes': graph.number_of_nodes(),
            'n_edges': graph.number_of_edges(),
            'density': nx.density(graph) if graph.number_of_nodes() > 0 else 0,
            'n_isolated': len([n for n in graph.nodes() if graph.degree(n) == 0]),
            'n_sources': len([n for n in graph.nodes() if graph.in_degree(n) == 0 and graph.out_degree(n) > 0]),
            'n_sinks': len([n for n in graph.nodes() if graph.out_degree(n) == 0 and graph.in_degree(n) > 0])
        }
        
        if graph.number_of_nodes() > 0:
            # Degree statistics
            in_degrees = [d for n, d in graph.in_degree()]
            out_degrees = [d for n, d in graph.out_degree()]
            
            stats.update({
                'avg_in_degree': np.mean(in_degrees),
                'max_in_degree': max(in_degrees),
                'avg_out_degree': np.mean(out_degrees),
                'max_out_degree': max(out_degrees)
            })
            
            # Find hubs
            in_degree_dict = dict(graph.in_degree())
            out_degree_dict = dict(graph.out_degree())
            
            # Top regulators (high out-degree)
            top_regulators = sorted(out_degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            stats['top_regulators'] = [(tf, deg) for tf, deg in top_regulators]
            
            # Top regulated (high in-degree)
            top_regulated = sorted(in_degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            stats['top_regulated'] = [(tf, deg) for tf, deg in top_regulated]
            
            # Check connectivity
            stats['is_weakly_connected'] = nx.is_weakly_connected(graph)
            if stats['is_weakly_connected']:
                stats['n_weakly_connected_components'] = 1
            else:
                stats['n_weakly_connected_components'] = nx.number_weakly_connected_components(graph)
        
        return stats
    
    def get_network_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all TFs with their scores and network properties.
        
        Returns:
            DataFrame with TF properties
        """
        if not self.results.get('network'):
            raise ValueError("Pipeline must be run first")
        
        graph = self.results['network']
        data = []
        
        for tf in graph.nodes():
            row = {
                'TF': tf,
                'is_identity': tf in self.results['identity_tfs'],
                'is_known': tf in self.known_identity_tfs,
                'in_degree': graph.in_degree(tf),
                'out_degree': graph.out_degree(tf),
                'total_degree': graph.degree(tf)
            }
            
            # Add scores if available
            if 'jsd_scores' in self.results:
                row['jsd_score'] = self.results['jsd_scores'].get(tf, 0)
            
            if 'centrality_scores' in self.results:
                row['centrality_score'] = self.results['centrality_scores'].get(tf, 0)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.sort_values('centrality_score', ascending=False) if 'centrality_score' in df.columns else df
    
    def visualize_network(self, save_path: Optional[str] = None, show_labels: bool = True):
        """
        Visualize the regulatory network.
        
        Args:
            save_path: Optional path to save figure
            show_labels: Whether to show TF labels
        """
        import matplotlib.pyplot as plt
        
        if not self.results.get('network'):
            raise ValueError("Pipeline must be run first")
        
        graph = self.results['network']
        identity_tfs = set(self.results['identity_tfs'])
        known_tfs = set(self.known_identity_tfs)
        
        # Create layout
        pos = nx.spring_layout(graph, seed=42, k=2, iterations=50)
        
        # Set up colors
        node_colors = []
        for node in graph.nodes():
            if node in identity_tfs and node in known_tfs:
                node_colors.append('red')  # Known and found
            elif node in identity_tfs:
                node_colors.append('orange')  # Found but not known
            elif node in known_tfs:
                node_colors.append('blue')  # Known but not found
            else:
                node_colors.append('lightgray')  # Others
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Draw network
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=300, alpha=0.7)
        nx.draw_networkx_edges(graph, pos, edge_color='gray', alpha=0.3, arrows=True)
        
        if show_labels:
            # Only label identity TFs and known TFs
            labels = {n: n for n in graph.nodes() if n in identity_tfs or n in known_tfs}
            nx.draw_networkx_labels(graph, pos, labels, font_size=8)
        
        plt.title(f"ChIP-seq Regulatory Network - {self.target_cell_type}")
        plt.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Known & Found'),
            Patch(facecolor='orange', label='Found (Novel)'),
            Patch(facecolor='blue', label='Known (Missed)'),
            Patch(facecolor='lightgray', label='Other TFs')
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()