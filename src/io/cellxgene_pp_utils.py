import cellxgene_census
import tiledbsoma as soma
from typing import List, Dict, Tuple, Any
import json
from pathlib import Path
import pandas as pd
from multiprocessing import Pool, cpu_count

class CellxgenePpUtils:
    def __init__(self, organism='homo_sapiens', obs_cols=None, default_obs_filter=None):
        """
        Initialize CellxgenePpUtils with flexible filtering options.
        
        Parameters:
        -----------
        organism : str
            Either 'homo_sapiens' or 'mus_musculus'
        obs_cols : dict, optional
            Custom observation columns configuration
        default_obs_filter : str, optional
            Default filter string to apply to all queries unless overridden
        """
        ## Experiment initializations 
        self.organism = organism
        self.census = cellxgene_census.open_soma()
        self.experiment = self.census["census_data"][self.organism]

        ## Default obs value filter and obs value columns 
        self.default_obs_cols = {
            "metadata_fields": [
                'suspension_type', 
                'tissue_general', 
                'sex_ontology_term_id', 
                'donor_id',
                'assay_ontology_term_id', 
                'self_reported_ethnicity_ontology_term_id',
                'tissue_ontology_term_id', 
                'disease_ontology_term_id',
                'development_stage_ontology_term_id', 
                'cell_type_ontology_term_id',
                'is_primary_data', 
                'cell_type', 
                'assay', 
                'disease', 
                'sex', 
                'tissue',
                'self_reported_ethnicity', 
                'development_stage', 
                'observation_joinid', 
                'dataset_id'
            ],
            "min_cells_per_ct": 50
        }
        
        # Set default obs filter
        self.default_obs_filter = default_obs_filter or "disease == 'normal' and is_primary_data == True"
        
        # Merge with custom obs_cols if provided
        self.obs_cols = self.default_obs_cols | (obs_cols or {})

    def build_filter(self, base_filter: Optional[str] = None, 
                     additional_filters: Optional[List[str]] = None,
                     use_default: bool = True) -> str:
        """
        Build a filter string by combining base filter, default filter, and additional filters.
        
        Parameters:
        -----------
        base_filter : str, optional
            A custom base filter to use (overrides default if use_default=False)
        additional_filters : list of str, optional
            Additional filter conditions to append with 'and'
        use_default : bool
            Whether to include the default filter (default: True)
        
        Returns:
        --------
        str : Combined filter string
        
        Examples:
        ---------
        # Use default filter only
        build_filter()
        
        # Use default + additional filters
        build_filter(additional_filters=["cell_type == 'B cell'", "tissue == 'blood'"])
        
        # Custom base filter without default
        build_filter(base_filter="disease == 'COVID-19'", use_default=False)
        
        # Combine everything
        build_filter(additional_filters=["sex == 'female'"])
        """
        filters = []
        
        # Add default filter if requested
        if use_default and self.default_obs_filter:
            filters.append(f"({self.default_obs_filter})")
        
        # Add base filter if provided (only if not using default)
        if base_filter and not use_default:
            filters.append(f"({base_filter})")
        
        # Add additional filters
        if additional_filters:
            for f in additional_filters:
                filters.append(f"({f})")
        
        # Combine all filters with 'and'
        return " and ".join(filters) if filters else None

    def list_datasets(self) -> Any:
        """List all datasets in the census."""
        datasets = self.census["census_info"]["datasets"].read().concat().to_pandas()
        return datasets
    
    def get_dataset_info(self, dataset_name: str) -> Any:
        """Get information about a specific dataset."""
        dataset_info = self.census.get_dataset_info(dataset_name)
        return dataset_info
    
    def get_ct_for_multiple_tissues(self, 
                                     ct: str, 
                                     tissue_list: List[str], 
                                     get_adata: bool = False,
                                     custom_filter: Optional[str] = None,
                                     additional_filters: Optional[List[str]] = None,
                                     use_default_filter: bool = True) -> Any:
        """
        Query for a specific cell type across multiple tissues.
        
        Parameters:
        -----------
        ct : str
            Cell type to query
        tissue_list : list of str
            List of tissues to include
        get_adata : bool
            Whether to return AnnData object (default: False returns query)
        custom_filter : str, optional
            Custom filter to use instead of default
        additional_filters : list of str, optional
            Additional filter conditions to add
        use_default_filter : bool
            Whether to use the default filter (default: True)
        
        Returns:
        --------
        Query object or AnnData depending on get_adata parameter
        """
        # Build the tissue and cell type filter
        base_conditions = [f"cell_type == '{ct}'", f"tissue in {tissue_list}"]
        
        # Combine with default or custom filters
        if custom_filter:
            filter_str = self.build_filter(
                base_filter=custom_filter, 
                additional_filters=base_conditions,
                use_default=False
            )
        else:
            filter_str = self.build_filter(
                additional_filters=base_conditions + (additional_filters or []),
                use_default=use_default_filter
            )
        
        query = self.experiment.axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=filter_str)
        )
        
        if get_adata:
            adata = query.to_anndata(
                X_name="raw", 
                column_names={"obs": self.obs_cols["metadata_fields"]}
            )
            return adata
        else:
            return query
        
    def get_multiple_tissues(self, 
                            tissue_list: List[str], 
                            get_adata: bool = False,
                            custom_filter: Optional[str] = None,
                            additional_filters: Optional[List[str]] = None,
                            use_default_filter: bool = True) -> Any:
        """
        Query for multiple tissues.
        
        Parameters:
        -----------
        tissue_list : list of str
            List of tissues to include
        get_adata : bool
            Whether to return AnnData object (default: False returns query)
        custom_filter : str, optional
            Custom filter to use instead of default
        additional_filters : list of str, optional
            Additional filter conditions to add
        use_default_filter : bool
            Whether to use the default filter (default: True)
        
        Returns:
        --------
        Query object or AnnData depending on get_adata parameter
        """
        base_conditions = [f"tissue in {tissue_list}"]
        
        if custom_filter:
            filter_str = self.build_filter(
                base_filter=custom_filter, 
                additional_filters=base_conditions,
                use_default=False
            )
        else:
            filter_str = self.build_filter(
                additional_filters=base_conditions + (additional_filters or []),
                use_default=use_default_filter
            )
        
        query = self.experiment.axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=filter_str)
        )
        
        if get_adata:
            adata = query.to_anndata(
                X_name="raw", 
                column_names={"obs": self.obs_cols["metadata_fields"]}
            )
            return adata
        else:
            return query

    def get_by_dataset_ids(self, 
                          dataset_ids: List[str], 
                          get_adata: bool = False,
                          custom_filter: Optional[str] = None,
                          additional_filters: Optional[List[str]] = None,
                          use_default_filter: bool = True) -> Any:
        """
        Query by dataset IDs.
        
        Parameters:
        -----------
        dataset_ids : list of str
            List of dataset IDs to include
        get_adata : bool
            Whether to return AnnData object (default: False returns query)
        custom_filter : str, optional
            Custom filter to use instead of default
        additional_filters : list of str, optional
            Additional filter conditions to add
        use_default_filter : bool
            Whether to use the default filter (default: True)
        
        Returns:
        --------
        Query object or AnnData depending on get_adata parameter
        """
        base_conditions = [f"dataset_id in {dataset_ids}"]
        
        if custom_filter:
            filter_str = self.build_filter(
                base_filter=custom_filter, 
                additional_filters=base_conditions,
                use_default=False
            )
        else:
            filter_str = self.build_filter(
                additional_filters=base_conditions + (additional_filters or []),
                use_default=use_default_filter
            )
        
        query = self.experiment.axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=filter_str)
        )

        if get_adata:
            adata = query.to_anndata(
                X_name="raw", 
                column_names={"obs": self.obs_cols["metadata_fields"]}
            )
            return adata
        else:
            return query
        
    def get_custom_dataset_complex_filter(self, 
                                         filter_str: str,
                                         get_adata: bool = False,
                                         use_default_filter: bool = False) -> Any:
        """
        Query with a completely custom complex filter.
        
        Parameters:
        -----------
        filter_str : str
            Custom filter string
        get_adata : bool
            Whether to return AnnData object (default: False returns query)
        use_default_filter : bool
            Whether to combine with default filter (default: False)
        
        Returns:
        --------
        Query object or AnnData depending on get_adata parameter
        """
        if use_default_filter:
            final_filter = self.build_filter(additional_filters=[filter_str])
        else:
            final_filter = filter_str
            
        query = self.experiment.axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=final_filter)
        )

        if get_adata:
            adata = query.to_anndata(
                X_name="raw", 
                column_names={"obs": self.obs_cols["metadata_fields"]}
            )
            return adata
        else:
            return query
    
    def get_by_cell_type(self,
                        cell_type: str,
                        get_adata: bool = False,
                        custom_filter: Optional[str] = None,
                        additional_filters: Optional[List[str]] = None,
                        use_default_filter: bool = True) -> Any:
        """
        Query by cell type.
        
        Parameters:
        -----------
        cell_type : str
            Cell type to query
        get_adata : bool
            Whether to return AnnData object (default: False returns query)
        custom_filter : str, optional
            Custom filter to use instead of default
        additional_filters : list of str, optional
            Additional filter conditions to add
        use_default_filter : bool
            Whether to use the default filter (default: True)
        
        Returns:
        --------
        Query object or AnnData depending on get_adata parameter
        """
        base_conditions = [f"cell_type == '{cell_type}'"]
        
        if custom_filter:
            filter_str = self.build_filter(
                base_filter=custom_filter, 
                additional_filters=base_conditions,
                use_default=False
            )
        else:
            filter_str = self.build_filter(
                additional_filters=base_conditions + (additional_filters or []),
                use_default=use_default_filter
            )
        
        query = self.experiment.axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=filter_str)
        )
        
        if get_adata:
            adata = query.to_anndata(
                X_name="raw", 
                column_names={"obs": self.obs_cols["metadata_fields"]}
            )
            return adata
        else:
            return query

    def update_default_filter(self, new_filter: str) -> None:
        """
        Update the default observation filter.
        
        Parameters:
        -----------
        new_filter : str
            New default filter string
        """
        self.default_obs_filter = new_filter
    
    def get_current_default_filter(self) -> str:
        """Get the current default filter."""
        return self.default_obs_filter


class MetadataBuilder:
    """
    Build metadata JSONs from CELLxGENE datasets
    """
    def __init__(self, output_dir: str = "./metadata"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.census_utils = CellxgenePpUtils(organism='homo_sapiens')
        self.datasets_info = []
    
    def get_organ_datasets(self, organ: str, n_datasets: int = 2) -> List[str]:
        """
        Find dataset IDs for a specific organ by querying tissue data
        
        Parameters:
        -----------
        organ : str
            Organ name (brain, blood, lung, pancreas, kidney)
        n_datasets : int
            Number of datasets to select for this organ
        
        Returns:
        --------
        List of dataset IDs
        """
        # Map our organ names to CELLxGENE tissue_general terms
        organ_to_tissue = {
            'brain': ['brain', 'nervous system'],
            'blood': ['blood', 'blood vessel', 'hematopoietic system'],
            'lung': ['lung', 'respiratory system'],
            'pancreas': ['pancreas'],
            'kidney': ['kidney']
        }
        
        tissue_terms = organ_to_tissue.get(organ, [organ])
        
        print(f"  Querying tissues: {tissue_terms}")
        
        # Query for these tissues (using first tissue term)
        query = self.census_utils.get_multiple_tissues(
            tissue_list=tissue_terms,
            get_adata=False,
            use_default_filter=True  # Gets normal, primary data
        )
        
        # Get obs dataframe to find dataset IDs
        obs_df = query.obs().concat().to_pandas()
        
        # Get unique dataset IDs and their cell counts
        dataset_counts = obs_df['dataset_id'].value_counts()
        
        # Select top N datasets by cell count
        top_datasets = dataset_counts.head(n_datasets).index.tolist()
        
        print(f"  Found {len(dataset_counts)} datasets, selecting top {len(top_datasets)}")
        for ds_id in top_datasets:
            print(f"    - {ds_id}: {dataset_counts[ds_id]:,} cells")
        
        return top_datasets
    
    def extract_metadata_from_dataset(self, 
                                    dataset_id: str,
                                    organ: str) -> Dict:
        """
        Query CELLxGENE and extract metadata for a single dataset
        Handles heterogeneous datasets (multiple donors, ages, etc.)
        """
        print(f"  Processing {dataset_id}...")
        
        try:
            # Query this specific dataset
            query = self.census_utils.get_by_dataset_ids(
                dataset_ids=[dataset_id],
                get_adata=False,
                use_default_filter=False
            )
            
            # Get the obs dataframe
            obs_df = query.obs().concat().to_pandas()
            
            if len(obs_df) == 0:
                raise ValueError(f"No data returned")
            
            # Clean data
            obs_df = obs_df.dropna(subset=['cell_type']).copy()
            
            if len(obs_df) < 10:
                raise ValueError(f"Too few valid cells ({len(obs_df)})")
            
            # === CHECK FOR HETEROGENEITY ===
            def check_heterogeneity(column_name):
                """Check if a column has multiple unique values"""
                if column_name not in obs_df.columns:
                    return "unknown", False
                
                unique_vals = obs_df[column_name].dropna().unique()
                
                if len(unique_vals) == 0:
                    return "unknown", False
                elif len(unique_vals) == 1:
                    return str(unique_vals[0]), False
                else:
                    # Multiple values - report as mixed
                    return f"mixed ({len(unique_vals)} values)", True
            
            # Check key metadata fields for heterogeneity
            tissue_val, tissue_mixed = check_heterogeneity('tissue')
            tissue_general_val, _ = check_heterogeneity('tissue_general')
            disease_val, disease_mixed = check_heterogeneity('disease')
            dev_stage_val, dev_stage_mixed = check_heterogeneity('development_stage')
            assay_val, assay_mixed = check_heterogeneity('assay')
            sex_val, sex_mixed = check_heterogeneity('sex')
            
            # Check donors
            n_donors = obs_df['donor_id'].nunique() if 'donor_id' in obs_df.columns else 1
            
            # Build cell types dict using groupby
            cell_types_dict = {}
            grouped = obs_df.groupby('cell_type', dropna=True)
            
            for ct_name, ct_group in grouped:
                if len(ct_group) == 0:
                    continue
                    
                first = ct_group.iloc[0]
                
                cell_types_dict[str(ct_name)] = {
                    "n_cells": len(ct_group),
                    "ontology_id": str(first.get('cell_type_ontology_term_id', 'unknown'))
                }
            
            if not cell_types_dict:
                raise ValueError("No cell types extracted")
            
            # Build metadata
            metadata = {
                "dataset_id": f"{organ}_{len(self.datasets_info) + 1:03d}",
                "source_id": dataset_id,
                
                "biological": {
                    "organism": "Homo sapiens",
                    "organ": organ,
                    "tissue": tissue_val,
                    "tissue_general": tissue_general_val,
                    "disease": disease_val,
                    "development_stage": dev_stage_val,
                    "sex": sex_val,
                    "n_donors": n_donors
                },
                
                "technical": {
                    "total_cells": len(obs_df),
                    "n_cell_types": len(cell_types_dict),
                    "assay": assay_val,
                    "suspension_type": str(obs_df.iloc[0].get('suspension_type', 'unknown'))
                },
                
                "heterogeneity_flags": {
                    "tissue_mixed": tissue_mixed,
                    "disease_mixed": disease_mixed,
                    "development_stage_mixed": dev_stage_mixed,
                    "assay_mixed": assay_mixed,
                    "sex_mixed": sex_mixed,
                    "multi_donor": n_donors > 1
                },
                
                "cell_types": cell_types_dict
            }
            
            # Print warnings for heterogeneous datasets
            if any([tissue_mixed, disease_mixed, dev_stage_mixed, assay_mixed, n_donors > 1]):
                print(f"    ⚠ Heterogeneous dataset detected:")
                if n_donors > 1:
                    print(f"      - {n_donors} donors")
                if dev_stage_mixed:
                    print(f"      - Multiple development stages")
                if disease_mixed:
                    print(f"      - Multiple disease states")
                if assay_mixed:
                    print(f"      - Multiple assays")
            
            return metadata
            
        except Exception as e:
            raise ValueError(f"Failed to extract metadata: {str(e)}")
    
    def create_dataset_metadata(self, dataset_id: str, organ: str) -> None:
        """
        Create and save metadata JSON for a single dataset
        """
        try:
            metadata = self.extract_metadata_from_dataset(dataset_id, organ)
            
            # Save to file
            filename = self.output_dir / f"{metadata['dataset_id']}.json"
            with open(filename, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Track for manifest
            self.datasets_info.append({
                "id": metadata['dataset_id'],
                "organ": organ,
                "cells": metadata['technical']['total_cells'],
                "n_cell_types": metadata['technical']['n_cell_types'],
                "source_id": dataset_id,
                "tissue": metadata['biological']['tissue']
            })
            
            print(f"    ✓ Saved {metadata['dataset_id']}: "
                  f"{metadata['technical']['total_cells']:,} cells, "
                  f"{metadata['technical']['n_cell_types']} cell types")
            
        except Exception as e:
            print(f"    ✗ Failed to process {dataset_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def create_manifest(self) -> None:
        """Create manifest.json summarizing all datasets"""
        manifest = {
            "version": "0.1.0",
            "date": "2025-11-20",
            "datasets": self.datasets_info,
            "stats": {
                "total_datasets": len(self.datasets_info),
                "total_cells": sum(d['cells'] for d in self.datasets_info),
                "total_cell_types": sum(d['n_cell_types'] for d in self.datasets_info),
                "organs": list(set(d['organ'] for d in self.datasets_info))
            }
        }
        
        with open(self.output_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ Manifest created: {len(self.datasets_info)} datasets")
        print(f"  Total cells: {manifest['stats']['total_cells']:,}")
        print(f"  Total cell types: {manifest['stats']['total_cell_types']}")
        print(f"  Organs: {', '.join(manifest['stats']['organs'])}")
    
    def build_repository(self, 
                        organs: List[str], 
                        datasets_per_organ: int = 2,
                        use_multiprocessing: bool = True,
                        n_processes: int = None):
        """
        Main method: build metadata for all organs
        
        Parameters:
        -----------
        organs : list of str
            List of organs to process
        datasets_per_organ : int
            Number of datasets per organ
        use_multiprocessing : bool
            Whether to use multiprocessing (default: True)
        n_processes : int
            Number of processes to use (default: number of CPUs)
        """
        print(f"Building repository for {len(organs)} organs...")
        print(f"Target: {datasets_per_organ} datasets per organ")
        
        if use_multiprocessing:
            if n_processes is None:
                n_processes = min(cpu_count(), len(organs))
            print(f"Using {n_processes} processes\n")
            
            # Prepare tasks: (organ, datasets_per_organ, output_dir)
            tasks = [(organ, datasets_per_organ, str(self.output_dir)) for organ in organs]
            
            # Use multiprocessing with standalone worker function
            with Pool(processes=n_processes) as pool:
                results = pool.map(_process_organ_worker, tasks)
            
            # Flatten results into datasets_info
            for organ_datasets in results:
                self.datasets_info.extend(organ_datasets)
                
        else:
            print("Using sequential processing\n")
            # Sequential processing
            for organ in organs:
                print(f"\n{'='*60}")
                print(f"=== {organ.upper()} ===")
                print('='*60)
                
                try:
                    dataset_ids = self.get_organ_datasets(organ, n_datasets=datasets_per_organ)
                    
                    if not dataset_ids:
                        print(f"  ⚠ No datasets found for {organ}")
                        continue
                    
                    for ds_id in dataset_ids:
                        self.create_dataset_metadata(ds_id, organ)
                        
                except Exception as e:
                    print(f"  ✗ Failed to process {organ}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Create manifest
        self.create_manifest()
        
        print("\n" + "="*60)
        print("DONE! Metadata created in:", self.output_dir)
        print("="*60)

# Standalone function for multiprocessing (must be at module level)
def _process_organ_worker(args: Tuple[str, int, str]) -> List[Dict]:
    """
    Worker function for multiprocessing - must be picklable
    
    Parameters:
    -----------
    args : tuple
        (organ, datasets_per_organ, output_dir)
    
    Returns:
    --------
    List of dataset info dictionaries
    """
    organ, datasets_per_organ, output_dir = args
    

    
    output_path = Path(output_dir)
    census_utils = CellxgenePpUtils(organism='homo_sapiens')
    organ_datasets = []
    
    print(f"\n{'='*60}")
    print(f"=== {organ.upper()} ===")
    print('='*60)
    
    try:
        # Map organ names to CELLxGENE tissue_general terms
        organ_to_tissue = {
            'brain': ['brain', 'nervous system'],
            'blood': ['blood', 'blood vessel', 'hematopoietic system'],
            'lung': ['lung', 'respiratory system'],
            'pancreas': ['pancreas'],
            'kidney': ['kidney']
        }
        
        tissue_terms = organ_to_tissue.get(organ, [organ])
        print(f"  Querying tissues: {tissue_terms}")
        
        # Query for these tissues
        query = census_utils.get_multiple_tissues(
            tissue_list=tissue_terms,
            get_adata=False,
            use_default_filter=True
        )
        
        # Get obs dataframe to find dataset IDs
        obs_df = query.obs().concat().to_pandas()
        dataset_counts = obs_df['dataset_id'].value_counts()
        top_datasets = dataset_counts.head(datasets_per_organ).index.tolist()
        
        print(f"  Found {len(dataset_counts)} datasets, selecting top {len(top_datasets)}")
        for ds_id in top_datasets:
            print(f"    - {ds_id}: {dataset_counts[ds_id]:,} cells")
        
        if not top_datasets:
            print(f"  ⚠ No datasets found for {organ}")
            return organ_datasets
        
        # Process each dataset
        for idx, ds_id in enumerate(top_datasets):
            try:
                # Query this specific dataset
                query = census_utils.get_by_dataset_ids(
                    dataset_ids=[ds_id],
                    get_adata=False,
                    use_default_filter=False
                )
                
                obs_df = query.obs().concat().to_pandas()
                
                if len(obs_df) == 0:
                    raise ValueError(f"No data returned")
                
                obs_df = obs_df.dropna(subset=['cell_type']).copy()
                
                if len(obs_df) < 10:
                    raise ValueError(f"Too few valid cells ({len(obs_df)})")
                
                # Check for heterogeneity
                def check_heterogeneity(column_name):
                    if column_name not in obs_df.columns:
                        return "unknown", False
                    unique_vals = obs_df[column_name].dropna().unique()
                    if len(unique_vals) == 0:
                        return "unknown", False
                    elif len(unique_vals) == 1:
                        return str(unique_vals[0]), False
                    else:
                        return f"mixed ({len(unique_vals)} values)", True
                
                tissue_val, tissue_mixed = check_heterogeneity('tissue')
                tissue_general_val, _ = check_heterogeneity('tissue_general')
                disease_val, disease_mixed = check_heterogeneity('disease')
                dev_stage_val, dev_stage_mixed = check_heterogeneity('development_stage')
                assay_val, assay_mixed = check_heterogeneity('assay')
                sex_val, sex_mixed = check_heterogeneity('sex')
                
                n_donors = obs_df['donor_id'].nunique() if 'donor_id' in obs_df.columns else 1
                
                # Build cell types dict
                cell_types_dict = {}
                grouped = obs_df.groupby('cell_type', dropna=True)
                
                for ct_name, ct_group in grouped:
                    if len(ct_group) == 0:
                        continue
                    first = ct_group.iloc[0]
                    cell_types_dict[str(ct_name)] = {
                        "n_cells": len(ct_group),
                        "ontology_id": str(first.get('cell_type_ontology_term_id', 'unknown'))
                    }
                
                if not cell_types_dict:
                    raise ValueError("No cell types extracted")
                
                # Build metadata
                metadata = {
                    "dataset_id": f"{organ}_{idx + 1:03d}",
                    "source_id": ds_id,
                    "biological": {
                        "organism": "Homo sapiens",
                        "organ": organ,
                        "tissue": tissue_val,
                        "tissue_general": tissue_general_val,
                        "disease": disease_val,
                        "development_stage": dev_stage_val,
                        "sex": sex_val,
                        "n_donors": n_donors
                    },
                    "technical": {
                        "total_cells": len(obs_df),
                        "n_cell_types": len(cell_types_dict),
                        "assay": assay_val,
                        "suspension_type": str(obs_df.iloc[0].get('suspension_type', 'unknown'))
                    },
                    "heterogeneity_flags": {
                        "tissue_mixed": tissue_mixed,
                        "disease_mixed": disease_mixed,
                        "development_stage_mixed": dev_stage_mixed,
                        "assay_mixed": assay_mixed,
                        "sex_mixed": sex_mixed,
                        "multi_donor": n_donors > 1
                    },
                    "cell_types": cell_types_dict
                }
                
                # Print warnings
                if any([tissue_mixed, disease_mixed, dev_stage_mixed, assay_mixed, n_donors > 1]):
                    print(f"    ⚠ Heterogeneous dataset detected:")
                    if n_donors > 1:
                        print(f"      - {n_donors} donors")
                    if dev_stage_mixed:
                        print(f"      - Multiple development stages")
                    if disease_mixed:
                        print(f"      - Multiple disease states")
                    if assay_mixed:
                        print(f"      - Multiple assays")
                
                # Save to file
                filename = output_path / f"{metadata['dataset_id']}.json"
                with open(filename, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Track for manifest
                dataset_info = {
                    "id": metadata['dataset_id'],
                    "organ": organ,
                    "cells": metadata['technical']['total_cells'],
                    "n_cell_types": metadata['technical']['n_cell_types'],
                    "source_id": ds_id,
                    "tissue": metadata['biological']['tissue']
                }
                organ_datasets.append(dataset_info)
                
                print(f"    ✓ Saved {metadata['dataset_id']}: "
                      f"{metadata['technical']['total_cells']:,} cells, "
                      f"{metadata['technical']['n_cell_types']} cell types")
                
            except Exception as e:
                print(f"    ✗ Failed to process {ds_id}: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"  ✗ Failed to process {organ}: {e}")
        import traceback
        traceback.print_exc()
    
    return organ_datasets
