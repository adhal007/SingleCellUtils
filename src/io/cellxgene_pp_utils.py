import cellxgene_census
import tiledbsoma as soma
from typing import Optional, List, Dict, Any

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

