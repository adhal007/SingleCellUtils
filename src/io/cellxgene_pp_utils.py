import cellxgene_census
import tiledbsoma as soma

class CellxgenePpUtils:
    def __init__(self, organism="human", params=None):
        self.organism = organism
        self.default_params = {
            "metadata_fields": [ 'suspension_type', 
       'tissue_general', 'sex_ontology_term_id', 'donor_id',
       'assay_ontology_term_id', 'self_reported_ethnicity_ontology_term_id',
       'tissue_ontology_term_id', 'disease_ontology_term_id',
       'development_stage_ontology_term_id', 'cell_type_ontology_term_id',
       'is_primary_data', 'cell_type', 'assay', 'disease', 'sex', 'tissue',
       'self_reported_ethnicity', 'development_stage', 'observation_joinid', 'dataset_id'],
            "min_cells_per_ct": 50
            
        }
        self.params = self.default_params | (params or {})
        self.census = cellxgene_census.open_soma()
        self.soma_ctx = soma.SOMAContext()
    
    def list_datasets(self):
        datasets = self.census["census_info"]["datasets"].read().concat().to_pandas()
        return datasets
    
    def get_dataset_info(self, dataset_name):
        dataset_info = self.census.get_dataset_info(dataset_name)
        return dataset_info
    
    def get_ct_for_multiple_tissues(self, ct, tissue_list, get_adata=False):
        experiment = self.census["census_data"][self.organism]
        query = experiment.axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=f"cell_type == '{ct}' and tissue in {tissue_list}")
        )
        if get_adata:
            adata = query.to_anndata(X_name="raw", column_names={"obs": self.params["metadata_fields"]})
            return adata
        else:
            return query

    def get_by_dataset_ids(self, dataset_ids, get_adata=False):
        experiment = self.census["census_data"][self.organism]
        query = experiment.axis_query(
            measurement_name="RNA",
            obs_query=soma.AxisQuery(value_filter=f"dataset_id in {dataset_ids}")
        )
        if get_adata:
            adata = query.to_anndata(X_name="raw", column_names={"obs": self.params["metadata_fields"]})
            return adata
        else:
            return query