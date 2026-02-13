"""Base dataset class for data handling."""

import numpy as np
import torch


class BaseDataset:
    """Base class for dataset handling and preprocessing."""

    def __init__(self):
        """Initialize base dataset."""
        pass

    def init_cat_clonelabel(self) -> None:
        """Initialize category and clone label mappings.
        
        Creates mappings between clones, clone labels, and categories (healthy/tumor).
        Sets up the following attributes:
        - n_cat: Number of categories
        - clonelabels: List of unique clone labels
        - n_clonelabels: Number of clone labels
        - cat2clonelabels: Mapping from category to clone labels
        - cat2clusters: Mapping from category to cluster indices
        - cluster2cat: Array mapping cluster index to category
        - clonelabel2clusters: Mapping from clone label to cluster indices
        """
        self.n_cat = len(np.unique(list(self.clonelabel2cat.values())))
        self.clonelabels = list(np.unique(self.cluster2clonelabel))
        self.n_clonelabels = len(np.unique(self.cluster2clonelabel))
        assert self.n_cat == 2, "Expected exactly 2 categories (healthy and tumor)"
        
        self.cat2clonelabels = {}
        self.cat2clonelabels['healthy'] = [
            clonelabel for i, clonelabel in enumerate(self.clonelabels)
            if self.clonelabel2cat[clonelabel] == 'healthy'
        ]
        self.cat2clonelabels['tumor'] = [
            clonelabel for i, clonelabel in enumerate(self.clonelabels)
            if self.clonelabel2cat[clonelabel] == 'tumor'
        ]

        self.cat2clusters = {}
        self.cat2clusters['healthy'] = [
            i for i, clonelabel in enumerate(self.cluster2clonelabel)
            if self.clonelabel2cat[clonelabel] == 'healthy'
        ]
        self.cat2clusters['tumor'] = [
            i for i, clonelabel in enumerate(self.cluster2clonelabel)
            if self.clonelabel2cat[clonelabel] == 'tumor'
        ]

        self.cluster2cat = np.zeros(len(self.cluster2clonelabel)).astype(str)
        for i, clonelabel in enumerate(self.cluster2clonelabel):
            self.cluster2cat[i] = self.clonelabel2cat[clonelabel]

        self.clonelabel2clusters = {}
        for clonelabel in self.clonelabels:
            self.clonelabel2clusters[clonelabel] = [
                i for i, cand_clonelabel in enumerate(self.cluster2clonelabel)
                if clonelabel == cand_clonelabel
            ]

    def get_fold_change_obs(self, DIC: dict) -> np.ndarray:
        """Calculate observed fold changes from data.
        
        Parameters
        ----------
        DIC : dict
            Dictionary containing drug response data with keys:
            - 'n_r': Drug response counts
            - 'n0_r': Baseline drug response counts
            - 'n_c': Control counts
            - 'n0_c': Baseline control counts
            - 'masks': Dictionary with masking information
            
        Returns
        -------
        np.ndarray
            Fold change array of shape (D, Ndrug) where D is number of drugs
            and Ndrug is number of patients
        """
        D = DIC['n_r'].shape[1]
        Ndrug = DIC['n_r'].shape[2]
        fold_change_obs = np.zeros((D, Ndrug))
        
        for patient_id in range(Ndrug):
            for drug_id in range(D):
                # Get number of replicates
                nb_replis = torch.sum(DIC['masks']['R'][:, drug_id, patient_id])
                nb_replis_c = torch.sum(DIC['masks']['C'][:, patient_id])
                
                patient_control_data = (
                    DIC['n0_c'] / DIC['n_c']
                )[:nb_replis_c, patient_id]

                # Calculate fold change in log space
                patient_drug_data = torch.log(
                    DIC['n0_r'] / DIC['n_r']
                )[:nb_replis, drug_id, patient_id]
                patient_control_data = torch.log(
                    DIC['n0_c'] / DIC['n_c']
                )[:nb_replis_c, patient_id]
                
                fold_change = patient_control_data.mean() - patient_drug_data.mean()
                fold_change_obs[drug_id, patient_id] = fold_change
                
        return fold_change_obs