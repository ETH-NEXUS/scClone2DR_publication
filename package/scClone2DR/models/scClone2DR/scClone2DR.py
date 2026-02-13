from ...utils import *
import numpy as np
import torch
from copy import deepcopy
import pyro
from torch.distributions import constraints
from pyro import poutine
import pyro.distributions as dist
import torch.distributions as tdist
from ...datasets import *
from ...trainers import * 
from ...resultanalysis import *
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
from torch.distributions import MultivariateNormal, LowRankMultivariateNormal
import h5py
from tqdm import tqdm
import os

class scClone2DR(Trainer, SimulatedData, RealData, ComputeStatistics):
    """single-cell Clone to Drug Response.

    Parameters
    ----------
    path_fastdrug
        Absolute path where the Fast Drug data is located in case real data will be used.
    path_rna
        Absolute path where the RNA data is located in case real data will be used.
    mode_nu
        Defines the way the preassay effect will be modeled. 
        One of:

        * ``'noise_correction'`` - Correct technical noise using GAMs
        * ``'fixed'`` - Do not correct for potential technical noise in the pharmacoscopy measurements.
    mode_theta
        Defines the way the overdispersion effect will be modeled.
        One of:

        * ``'no_overdispersion'`` - No overdispersion is considered
        * ``'equal'`` - The overdispersion parameter is the same for all samples and it is shared for RNA and pharmacoscopy measurements.
        * ``'shared'`` - The overdispersion parameters are the same for all samples and we use two different ones for RNA and pharmacoscopy measurements.
        * ``'not shared coupled'`` - The overdispersion for RNA data is one parameter shared across patients. For the pharmacoscopy measurements, the overdispersion parameter of a given sample is expressed as a sample specific scaling factor times the overdispersion parameter used for the RNA data.
        * ``'not shared decoupled'`` - The overdispersion for RNA data is one parameter shared across patients. For the pharmacoscopy measurements, all samples have a different overdispersion parameters.

    """

    # Constants for numerical stability
    EPSILON = 1e-6
    DEFAULT_TOTAL_COUNT = 100000
    DEFAULT_THETA_RNA = 40.0
    DEFAULT_THETA_FD = 1.0
    DEFAULT_VAR_F = 0.1
    
    def __init__(self,
                 path_fastdrug=None,
                 path_rna=None,
                 path_info_cohort=None,
                 mode_nu: str = "noise_correction",
                 mode_theta: str = "not shared decoupled",
                 type_guide: str = "full_MVN",
                 rank: int = 20):
        Trainer.__init__(self, type_guide=type_guide, rank=rank)
        SimulatedData.__init__(self)
        RealData.__init__(self, path_fastdrug=path_fastdrug, path_rna=path_rna, path_info_cohort=path_info_cohort)
        ComputeStatistics.__init__(self)
        self.mode_nu = mode_nu
        self.mode_theta = mode_theta        

        
    def get_optimized_guide(self, params: dict):
        """Construct optimized guide distribution from learned parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary containing learned guide parameters
            
        Returns
        -------
        Distribution
            Multivariate normal distribution for the guide
        """
        if self.type_guide == "full_MVN":
            loc = params['AutoMultivariateNormal.loc']
            # Define the scale_tril (lower triangular matrix)
            scale_tril = params['AutoMultivariateNormal.scale_tril']
            # Define the multivariate Gaussian distribution
            guide = MultivariateNormal(loc=torch.tensor(loc), scale_tril=torch.tensor(scale_tril))
        elif self.type_guide == "lowrank_MVN":
            loc = torch.as_tensor(params['AutoLowRankMultivariateNormal.loc'])
            cov_factor = torch.as_tensor(params['AutoLowRankMultivariateNormal.cov_factor'])  # Low-rank factor
            cov_diag = torch.as_tensor(params['AutoLowRankMultivariateNormal.scale'])  # Diagonal correction
            guide = LowRankMultivariateNormal(loc=loc, cov_factor=cov_factor, cov_diag=cov_diag)
        elif self.type_guide == "diagonal":
            loc = params['AutoDiagonalNormal.loc']
            scale = params['AutoDiagonalNormal.scale']  # Standard deviations
            # Define diagonal covariance matrix
            covariance_matrix = torch.diag(scale**2)
            # Define the multivariate Gaussian distribution
            guide = MultivariateNormal(loc=torch.tensor(loc), covariance_matrix=torch.tensor(covariance_matrix))
        return guide

        
    def get_mean_logscore(self, proportions, D, pi, nu_healthy):
        proportions = torch.tensor(proportions)
        pi = torch.tensor(pi)
        control_0 = (torch.sum(proportions[self.cat2clusters['healthy'], :], dim=0)).unsqueeze(0).repeat(D, 1) * nu_healthy
        control_t = torch.sum(proportions[self.cat2clusters['tumor'], :].unsqueeze(0).repeat(D, 1, 1), dim=1) * (1 - nu_healthy)
        drug_0 = (torch.sum(proportions[self.cat2clusters['healthy'], :].unsqueeze(0).repeat(D, 1, 1) * pi[:, self.cat2clusters['healthy'], :], dim=1)) * nu_healthy
        drug_t = torch.sum(proportions[self.cat2clusters['tumor'], :].unsqueeze(0).repeat(D, 1, 1) * pi[:, self.cat2clusters['tumor'], :], dim=1) * (1 - nu_healthy)
        return torch.log(control_t * (drug_0 + drug_t)) - torch.log((control_0 + control_t) * drug_t)

    def get_mean_fracMEL_control(self, proportions: torch.Tensor, C: int, nu_healthy: torch.Tensor) -> torch.Tensor:
        """
        Compute the fraction of tumor cells in control wells.
        
        Parameters
        ----------
        proportions : torch.Tensor
            Clone proportions (Kmax x N)
        C : int
            Number of control wells
        nu_healthy : torch.Tensor
            Fraction of healthy cells in control wells
            
        Returns
        -------
        torch.Tensor
            Fraction of tumor cells in control wells (C x N)
        """
        proportions = torch.tensor(proportions)
        control_0 = (torch.sum(proportions[self.cat2clusters['healthy'], :], dim=0)).unsqueeze(0).repeat(C, 1) * nu_healthy
        control_t = torch.sum(proportions[self.cat2clusters['tumor'], :].unsqueeze(0).repeat(C, 1, 1), dim=1) * (1 - nu_healthy)
        return control_t / (control_0 + control_t)

    def get_mean_fracMEL_treated(self, proportions: torch.Tensor, D: int, pi: torch.Tensor, nu_healthy: torch.Tensor) -> torch.Tensor:
        """
        Compute the fraction of tumor cells in treated wells.
        
        Parameters
        ----------
        proportions : torch.Tensor
            Clone proportions (Kmax x N)
        D : int
            Number of drugs
        pi : torch.Tensor
            Survival probabilities (D x Kmax x N)
        nu_healthy : torch.Tensor
            Fraction of healthy cells in treated wells
            
        Returns
        -------
        torch.Tensor
            Fraction of tumor cells in treated wells (D x N)
        """
        proportions = torch.tensor(proportions)
        pi = torch.tensor(pi)
        control_0 = (torch.sum(proportions[self.cat2clusters['healthy'], :], dim=0)).unsqueeze(0).repeat(D, 1) * nu_healthy
        control_t = torch.sum(proportions[self.cat2clusters['tumor'], :].unsqueeze(0).repeat(D, 1, 1), dim=1) * (1 - nu_healthy)
        drug_0 = (torch.sum(proportions[self.cat2clusters['healthy'], :].unsqueeze(0).repeat(D, 1, 1) * pi[:, self.cat2clusters['healthy'], :], dim=1)) * nu_healthy
        drug_t = torch.sum(proportions[self.cat2clusters['tumor'], :].unsqueeze(0).repeat(D, 1, 1) * pi[:, self.cat2clusters['tumor'], :], dim=1) * (1 - nu_healthy)
        return drug_t / (drug_0 + drug_t)

    def compute_survival_probas_single_cell_features(self, data, params):
        """
        Compute the survival probability at the clone level from features at the single cell level using the attention mechanism.
        """
        Xsubclones = {}
        all_pis = []
        for clonelabel in self.clonelabels:
            idxs = self.clonelabel2clusters[clonelabel]
            offset_clonelabel = params.get('offset_{0}'.format(clonelabel))
            gamma_clonelabel = params.get('gamma_{0}'.format(clonelabel))
            Xsubclones[clonelabel] = torch.sum(
                data['X'][idxs, :, :, :] * (masked_softmax(
                    torch.matmul(data['X'][idxs, :, :, :], gamma_clonelabel),
                    data['masks']['SingleCell'][idxs, :, :],
                    dim=2
                )).unsqueeze(3),
                dim=2
            )
            all_pis.append((sigmoid(
                torch.matmul(Xsubclones[clonelabel], params['beta'].T) + offset_clonelabel.unsqueeze(0).unsqueeze(0)
            )).permute(2, 0, 1))  # D x len(idxs) x N
        pi = torch.cat(all_pis, dim=1)
        return pi
    
    def compute_survival_probas_subclone_features(self, data, params):
        """
        Compute the survival probability at the clone level from features at the subclone level.
        """
        all_pis = []
        for clonelabel in self.clonelabels:
            cat = self.clonelabel2cat[clonelabel]
            idxs = self.cat2clusters[cat]
            offset_clonelabel = params.get('offset_{0}'.format(clonelabel))
            all_pis.append((sigmoid(
                torch.matmul(data['X'][idxs, :, :], params['beta'].T) + offset_clonelabel.unsqueeze(0).unsqueeze(0)
            )).permute(2, 0, 1))  # D x len(idxs) x N
        pi = torch.cat(all_pis, dim=1)
        return pi
        
    def prior(self, data_prior, fixed_proportions=False, theta_rna=None):
        """
        Prior used in the SVI algorithm.

        Quantities of interest
        ----------------------
        N : Total number of patients
        Kmax : Maximum number of subclone per patient
        C : Total number of control wells (the same for each patient)
        R : Total number of wells with drug (the same for each patient)
        """
        Kmax = data_prior['Kmax']
        R, D, Ndrug = data_prior['n_r'].shape
        C, N = data_prior['n_c'].shape

        masks_train = data_prior['masks']
        dim = data_prior['X'].shape[-1]
        beta = pyro.param('beta', torch.zeros((D, dim)))
        

        params_pi = {"beta": beta}
        for clonelabel in self.clonelabels:
            params_pi["offset_{0}".format(clonelabel)] = pyro.param(
                "offset_{0}".format(clonelabel), torch.zeros(D)
            )
        if data_prior['single_cell_features']:
            var_fs = {}
            for clonelabel in self.clonelabels:
                var_fs[clonelabel] = pyro.param(
                    "f_{0}".format(clonelabel),
                    torch.tensor(self.DEFAULT_VAR_F),
                    constraints.positive
                )
                params_pi['gamma_{0}'.format(clonelabel)] = pyro.sample(
                    "gamma_{0}".format(clonelabel),
                    dist.Normal(torch.zeros(dim), var_fs[clonelabel] * torch.ones(dim)).to_event(1)
                )
            pi = self.compute_survival_probas_single_cell_features(data_prior, params_pi)
        else:
            pi = self.compute_survival_probas_subclone_features(data_prior, params_pi)
            
            
        if theta_rna is None:
            theta_rna = pyro.param('theta_rna', torch.tensor(self.DEFAULT_THETA_RNA), constraints.positive)

        try:
            weights = data_prior['weights']
        except:
            weights = torch.ones(N)

        try:
            weights_ND = data_prior['weights_ND']
        except:
            weights_ND = torch.ones((D, N))


        if self.mode_nu == "noise_correction":
            dim_c = data_prior['X_nu_control'].shape[2]
            beta_control = pyro.param('beta_control', torch.zeros(dim_c))
        
        with pyro.plate('samples', N):
            with pyro.poutine.scale(scale=weights):
                if fixed_proportions:
                    proportions = (data_prior['n_rna'] / torch.tile(
                        torch.sum(data_prior['n_rna'], dim=0).reshape(1, N), (Kmax, 1)
                    ))
                else:
                    proportions = pyro.param(
                        'proportions', data_prior['ini_proportions'].clone(), constraints.simplex
                    ).T


                # OVERDISPERSION PARAMETER (One for each patient)
                if self.mode_theta == 'equal':
                    theta_fd = theta_rna
                elif self.mode_theta == 'shared':
                    theta_fd = pyro.param('theta_fd', torch.tensor(self.DEFAULT_THETA_FD), constraints.positive)
                elif 'not shared' in self.mode_theta:
                    theta_fd = pyro.param('theta_fd', self.DEFAULT_THETA_RNA * torch.ones(N), constraints.positive)

                # RNA DATA
                # When calculating the log probability Multinomial distribution ignores total counts argument (it calculates it from the values). 
                # Since you are not sampling from the Multinomial distribution here, we can safely just set total_count to an arbitrary value
                if not(data_prior['n_rna'] is None): # for the bulk model
                    if self.mode_theta == "no_overdispersion":
                        n_rna = pyro.sample('n_rna', dist.Multinomial(self.DEFAULT_TOTAL_COUNT, proportions.T), obs=data_prior['n_rna'].T)
                    elif self.mode_theta in ['equal', 'shared', 'not shared decoupled']:
                        n_rna = pyro.sample('n_rna', dist.DirichletMultinomial((theta_rna * proportions).T, torch.sum(data_prior['n_rna'], dim=0)), obs=data_prior['n_rna'].T)
                    else:
                        n_rna = pyro.sample('n_rna', dist.DirichletMultinomial(( theta_rna * theta_fd * proportions).T, torch.sum(data_prior['n_rna'], dim=0)), obs=data_prior['n_rna'].T)

                # CONTROL WELLS
                with pyro.plate('controls', C), poutine.mask(mask=masks_train['C']):

                    if self.mode_nu == "fixed":
                        nu_tumor_over_nu_healthy = torch.tensor(1)
                        nu_healthy = 1. / (1 + nu_tumor_over_nu_healthy)

                    elif self.mode_nu == "noise_correction":
                        nu_tumor_over_nu_healthy = torch.exp(data_prior['X_nu_control'] @ beta_control)
                        nu_healthy = 1. / (1 + nu_tumor_over_nu_healthy)

                    nu_tumor = 1 - nu_healthy

                    if self.mode_theta == "no_overdispersion":
                        n0_c = pyro.sample(
                            'n0_c',
                            dist.Binomial(
                                self.DEFAULT_TOTAL_COUNT,
                                (proportions[0, :]).unsqueeze(0).repeat(C, 1) * nu_healthy
                            ),
                            obs=data_prior['n0_c']
                        )
                    else:
                        n0_c = pyro.sample(
                            'n0_c',
                            dist.BetaBinomial(
                                (theta_fd * torch.sum(proportions[self.cat2clusters['healthy'], :], dim=0)).unsqueeze(0).repeat(C, 1) * nu_healthy,
                                (theta_fd * torch.sum(proportions[self.cat2clusters['tumor'], :], dim=0)).unsqueeze(0).repeat(C, 1) * (1 - nu_healthy),
                                data_prior['n_c']
                            ),
                            obs=data_prior['n0_c']
                        )
                    

        with pyro.plate('samples_drug', Ndrug):

            # WELLS WITH DRUG
            with pyro.plate('drugs', D):
                with pyro.plate('replicates', R), poutine.mask(mask=masks_train['R']):
                    if self.mode_nu == "fixed":
                        nu_tumor_over_nu_healthy = torch.tensor(1)
                        nu_healthy_drug = 1. / (1 + nu_tumor_over_nu_healthy)
                    elif self.mode_nu == "noise_correction":
                        nu_tumor_over_nu_healthy = torch.exp(data_prior['X_nu_drug'] @ beta_control)
                        nu_healthy_drug = 1. / (1 + nu_tumor_over_nu_healthy)
                    nu_tumor_drug = 1 - nu_healthy_drug

                    if 'not' in self.mode_theta:
                        theta_fd_mode = theta_fd[:Ndrug]
                    else:
                        theta_fd_mode = theta_fd

                    if self.mode_theta == "no_overdispersion":
                        n0_r = pyro.sample(
                            'n0_r',
                            dist.Binomial(
                                self.DEFAULT_TOTAL_COUNT,
                                (torch.sum(
                                    proportions[self.cat2clusters['healthy'], :Ndrug].unsqueeze(0).repeat(D, 1, 1) * pi[:, self.cat2clusters['healthy'], :],
                                    dim=1
                                )).unsqueeze(0).repeat(R, 1, 1) * nu_healthy_drug
                            ),
                            obs=data_prior['n0_r']
                        )
                    else:
                        n0_r = pyro.sample(
                            'n0_r',
                            dist.BetaBinomial(
                                (theta_fd_mode * torch.sum(
                                    proportions[self.cat2clusters['healthy'], :Ndrug].unsqueeze(0).repeat(D, 1, 1) * pi[:, self.cat2clusters['healthy'], :],
                                    dim=1
                                )).unsqueeze(0).repeat(R, 1, 1) * nu_healthy_drug,
                                (theta_fd_mode * torch.sum(
                                    proportions[self.cat2clusters['tumor'], :Ndrug].unsqueeze(0).repeat(D, 1, 1) * pi[:, self.cat2clusters['tumor'], :],
                                    dim=1
                                )).unsqueeze(0).repeat(R, 1, 1) * (1 - nu_healthy_drug),
                                data_prior['n_r']
                            ),
                            obs=data_prior['n0_r']
                        )
                    frac_r = 1. - n0_r / data_prior['n_r']

        
    def convert_to_tensor(self, params: dict) -> dict:
        """
        Convert numpy arrays in the dictionary provided as input as Pytorch tensors.
        
        Parameters
        ----------
        params : dict
            Dictionary potentially containing numpy arrays
            
        Returns
        -------
        dict
            Dictionary with numpy arrays converted to torch tensors
        """
        if not (params is None):
            for key, val in params.items():
                if isinstance(val, np.ndarray):  # Check if the value is a numpy array
                    try:
                        params[key] = torch.tensor(val)
                    except:
                        params[key] = val
        return params
    
    def get_survival_probas(self, data):
        """
        Compute the survival probabilities
        """
        params_pi = {"beta": data['beta']}
        for clonelabel in self.clonelabels:
            params_pi["offset_{0}".format(clonelabel)] = data["offset_{0}".format(clonelabel)]
        if data['single_cell_features']:
            for clonelabel in self.clonelabels:
                params_pi['gamma_{0}'.format(clonelabel)] = data["gamma_{0}".format(clonelabel)]
            pi = self.compute_survival_probas_single_cell_features(data, params_pi)
        else:
            pi = self.compute_survival_probas_subclone_features(data, params_pi)
        return pi

    
    def sampling(self, dic, params=None):
        """
        Use the generative model to sample new data using the learned parameters.
        If features at the cell level are considered, params should contain parameters 'gamma_{clonelabel}' for all clonelabels. 
        """
        data = self.merge_data_param(dic, params)
        N,Kmax,R,C,D = data['N'],data['Kmax'],data['R'],data['C'],data['D']
        R,D,Ndrug = data['n_r'].shape
        beta = data['beta']
        masks = data['masks']
                    
        pi = self.get_survival_probas(data)
            
        proportions = (data['proportions'] / torch.sum(data['proportions'], dim=1).unsqueeze(1)).T
        proportions = proportions[:,-N:]

        if self.mode_nu == "noise_correction":
            beta_control = data['beta_control']

        if self.mode_theta == 'equal':
            theta_fd = data['theta_rna']
        else:
            theta_fd = data['theta_fd']

        theta_rna = data['theta_rna']

        n_rna = np.zeros((Kmax, N))
        if data['n_rna'] is not None:
            for i in range(N):
                idxs_notnull = np.where(proportions[:, i] > 0)[0]
                if self.mode_theta in ['equal', 'shared', 'not shared decoupled']:
                    n_rna[idxs_notnull, i] = dist.DirichletMultinomial(
                        theta_rna * proportions[idxs_notnull, i],
                        torch.sum(data['n_rna'][idxs_notnull, i])
                    ).sample()
                elif self.mode_theta == 'not shared coupled':
                    n_rna[idxs_notnull, i] = dist.DirichletMultinomial(
                        theta_rna * theta_fd[i] * proportions[idxs_notnull, i],
                        torch.sum(data['n_rna'][idxs_notnull, i])
                    ).sample()
        else:
            n_rna = None

        with pyro.plate('samples', N):
            # CONTROL WELLS
            with pyro.plate('controls', C), poutine.mask(mask=masks['C']):
                if self.mode_nu == "fixed":
                    nu_tumor_over_nu_healthy = torch.tensor(1)
                    nu_healthy_c = 1. / (1 + nu_tumor_over_nu_healthy)
                elif self.mode_nu == "noise_correction":
                    nu_tumor_over_nu_healthy = torch.exp(data['X_nu_control'] @ beta_control)
                    nu_healthy_c = 1. / (1 + nu_tumor_over_nu_healthy)

                nu_tumor_c = 1 - nu_healthy_c

                n0_c = pyro.sample(
                    'n0_c',
                    dist.BetaBinomial(
                        (theta_fd * torch.sum(proportions[self.cat2clusters['healthy'], :], dim=0)).unsqueeze(0).repeat(C, 1) * nu_healthy_c,
                        (theta_fd * torch.sum(proportions[self.cat2clusters['tumor'], :], dim=0)).unsqueeze(0).repeat(C, 1) * (1 - nu_healthy_c),
                        data['n_c']
                    )
                )

                frac_c = 1. - n0_c / data['n_c']

        with pyro.plate('samples', Ndrug):

            # WELLS WITH DRUG
            with pyro.plate('drugs', D):
                with pyro.plate('replicates', R), poutine.mask(mask=masks['R']):
                    if self.mode_nu == "fixed":
                        nu_tumor_over_nu_healthy = torch.tensor(1)
                        nu_healthy_drug = 1. / (1 + nu_tumor_over_nu_healthy)
                    elif self.mode_nu == "noise_correction":
                        nu_tumor_over_nu_healthy = torch.exp(data['X_nu_drug'] @ beta_control)
                        nu_healthy_drug = 1. / (1 + nu_tumor_over_nu_healthy)

                    nu_tumor_drug = 1 - nu_healthy_drug

                    if 'not' in self.mode_theta:
                        theta_fd_mode = theta_fd[:Ndrug]
                    else:
                        theta_fd_mode = theta_fd
                    n0_r = pyro.sample(
                        'n0_r',
                        dist.BetaBinomial(
                            (theta_fd_mode * torch.sum(
                                proportions[self.cat2clusters['healthy'], :Ndrug].unsqueeze(0).repeat(D, 1, 1) * pi[:, self.cat2clusters['healthy'], :],
                                dim=1
                            )).unsqueeze(0).repeat(R, 1, 1) * nu_healthy_drug,
                            (theta_fd_mode * torch.sum(
                                proportions[self.cat2clusters['tumor'], :Ndrug].unsqueeze(0).repeat(D, 1, 1) * pi[:, self.cat2clusters['tumor'], :],
                                dim=1
                            )).unsqueeze(0).repeat(R, 1, 1) * (1 - nu_healthy_drug),
                            data['n_r']
                        )
                    )
                    frac_r = 1. - n0_r / data['n_r']

                frac_mean_r = torch.sum(masks['R'] * torch.nan_to_num(frac_r), dim=0) / torch.sum(masks['R'], dim=0)
                frac_mean_c = torch.sum(masks['C'] * torch.nan_to_num(frac_c), dim=0) / torch.sum(masks['C'], dim=0)

                log_score = torch.log(frac_mean_r) - torch.log(frac_mean_c)[:Ndrug]

        return {
            'n0_c': n0_c, 'n0_r': n0_r, 'n_rna': n_rna, 'frac_r': frac_r, 'frac_c': frac_c,
            'frac_mean_r': frac_mean_r, 'frac_mean_c': frac_mean_c, 'log_score': log_score,
            'pi': pi, 'nu_healthy_drug': nu_healthy_drug, 'nu_healthy_control': nu_healthy_c,
            'N': N, 'Kmax': Kmax, 'R': R, 'C': C, 'D': D
        }        


    def get_posterior_mean_latentvar(self, params, nsamples=100):
        postmeans = {}
        guide = self.get_optimized_guide(params)
        def sample_latent_variables():
            return guide.sample()
            
        samp_gamma = sample_latent_variables()
        dim = len(samp_gamma)//self.n_clonelabels
        for ite in range(nsamples):
            for i, clonelabel in enumerate(self.clonelabels):
                if ite==0:
                    postmeans['gamma_{0}'.format(clonelabel)] = samp_gamma[dim*i:dim*(i+1)] / nsamples
                else:
                    postmeans['gamma_{0}'.format(clonelabel)] += samp_gamma[dim*i:dim*(i+1)] / nsamples
        return postmeans
    
    def sampling_from_posterior(self, data, dir_save, params=None, nb_ites=100, sample_names=None, model_name=''):
        """
        Use the generative model to sample new data using the learned parameters.
        This method is only useful when features are defined at the cell level. We sample from the posterior distribution the parameters 'gamma_{clonelabel}' for all clonelabels in order to compute the posterior mean and stds of the different statistics of interest.
        """
        if sample_names is None:
            sample_names = [i for i in range(data['N'])]
        ### Dealing with the posterior
        N, D, R, Kmax = data['N'], data['D'], data['R'], data['Kmax']
        
        ## data_sample will be used to store the posterior means of the different model parameters.
        data_sample = deepcopy(data)
        for key, val in params.items():
            data_sample[key] = torch.tensor(val)
        data_sample['proportions'] = data_sample['proportions']
        if self.mode_theta != "equal":
            data_sample['theta_fd'] = data_sample['theta_fd']
        
        ## data_esti_ghost contains the observed data and the learned model parameters. It will be used to sample from the posterior.
        data_esti_ghost = deepcopy(data)
        for key, val in params.items():
            data_esti_ghost[key] = torch.tensor(val)
        data_esti_ghost['proportions'] = data_esti_ghost['proportions']
        if self.mode_theta != "equal":
            data_esti_ghost['theta_fd'] = data_esti_ghost['theta_fd']
            
            
        try:
            guide = self.guide
            def sample_latent_variables():
                return guide.sample_latent()
        except:
            guide = self.get_optimized_guide(params)
            def sample_latent_variables():
                return guide.sample()
            
        ## We initialize the variables of interest to zeros in data_sample
        samp_gamma = sample_latent_variables()
        dim = len(samp_gamma)//self.n_clonelabels
        for i, clonelabel in enumerate(self.clonelabels):
            data_esti_ghost['gamma_{0}'.format(clonelabel)] = samp_gamma[dim*i:dim*(i+1)]
        data_sample_ghost = self.sampling(data_esti_ghost)
        for key,val in data_sample_ghost.items():
            data_sample[key] = val
        ls_var_2_avg = ['n0_c', 'n0_r',  'n_rna', 'frac_r', 'frac_c', 'frac_mean_r', 'frac_mean_c', 'log_score', 'pi','nu_healthy_drug', 'nu_healthy_control']
        for var in ls_var_2_avg:
            if not(data_sample_ghost[var] is None):
                data_sample[var] = torch.zeros(data_sample_ghost[var].shape)


        PI = torch.zeros((nb_ites,D,Kmax,N))
        latent_dim = params['beta'].shape[1]
        PI = torch.zeros((D,Kmax,N))
        LRR = torch.zeros((D,Kmax,N,latent_dim))
        LOR = torch.zeros((D,Kmax,N,latent_dim))
        ME = torch.zeros((D,Kmax,N,latent_dim))
        subclone_features = torch.zeros((Kmax,N,latent_dim))
        with torch.no_grad():
            for i in tqdm(range(nb_ites)):
                samp_gamma = sample_latent_variables()
                for j, clonelabel in enumerate(self.clonelabels):
                    data_esti_ghost['gamma_{0}'.format(clonelabel)] = samp_gamma[dim*j:dim*(j+1)]
                data_sample_ghost = self.sampling(data_esti_ghost)
                pi = self.compute_survival_probas_single_cell_features(data, data_esti_ghost)
                PI += pi.detach()/nb_ites

                for clonelabel in self.clonelabels:
                    idxs = self.clonelabel2clusters[clonelabel]
                    subclone_features[idxs,:,:] += torch.sum(data['X'][idxs,:,:,:] * (masked_softmax(torch.matmul(data['X'][idxs,:,:,:], data_esti_ghost['gamma_{0}'.format(clonelabel)]), data['masks']['SingleCell'][idxs,:,:], dim=2)).unsqueeze(3), dim=2) /nb_ites


                for var in ls_var_2_avg:
                    if not(data_sample_ghost[var] is None):
                        data_sample[var] += data_sample_ghost[var]/nb_ites
                ## END: Dealing with posterior

                for j in range(latent_dim):
                    params_pi = {}
                    for l, clonelabel in enumerate(self.clonelabels):
                        params_pi["gamma_{0}".format(clonelabel)] = samp_gamma[dim*l:dim*(l+1)]
                        params_pi["offset_{0}".format(clonelabel)] = torch.tensor(params["offset_{0}".format(clonelabel)])
                    beta_j = deepcopy(params['beta'])                
                    beta_j[:,j] = 0
                    params_pi["beta"] = torch.tensor(beta_j)
                    pi_j = self.compute_survival_probas_single_cell_features(data, params_pi)

                    # LOR
                    LOR[:,:,:,j] += (torch.log(pi/(1-pi))  -  torch.log(pi_j/(1-pi_j)))/nb_ites

                    # LRR
                    LRR[:,:,:,j] += torch.log(pi/pi_j)/nb_ites

                # ME: D x Kmax x N x dim
                ME += (data_sample_ghost['pi'] * (1-data_sample_ghost['pi']))[:,:,:,None] * params['beta'][:,None,None,:] /nb_ites


        MASK = data['masks']['RNA']
        if np.sum(MASK.detach().numpy())==0:
            MASK = torch.sum(data['masks']['SingleCell'], dim=2)>0.5
            
        subclone_features = subclone_features.detach().numpy()
        for j in range(latent_dim):
            subclone_features[:,:,j][~MASK] = float('nan')
        all_local_importances = params['beta'][:,None,None,:] * subclone_features[None,:,:,:]
        with h5py.File(os.path.join(dir_save,model_name+'local_importance.h5'), 'w') as f:
            # Create a dataset
            dset = f.create_dataset('local_importance_mean', data=all_local_importances)

            column_names = {
            'dim2_subclones': [i for i in range(data['Kmax'])],
            'dim3_samples': sample_names, #np.char.encode(sample_names_test, 'ascii'),
            'dim4_dimensions': self.feature_names,
            'dim1_drugs':  self.FD.selected_drugs
            }

            for dim, labels in column_names.items():
                dset.attrs[dim] = labels


        with h5py.File(os.path.join(dir_save,model_name+'subclone_features.h5'), 'w') as f:
            # Create a dataset
            dset = f.create_dataset('subclone_features_posterior_mean', data=subclone_features)
            column_names = {
            'dim1_subclones': [i for i in range(data['Kmax'])],
            'dim2_samples': sample_names, # np.char.encode(sample_names, 'ascii'),
            'dim3_dimensions': self.feature_names
            }

            for dim, labels in column_names.items():
                dset.attrs[dim] = labels


        postmean = PI.numpy()
        for d in range(data['D']):
            postmean[d,:,:][~MASK] = float('nan')
        with h5py.File(os.path.join(dir_save,model_name+'survival_probabilities.h5'), 'w') as f:
            # Create a dataset
            dset = f.create_dataset('survival_probabilities_posterior_mean', data=postmean)
            column_names = {
            'dim2_subclones': [i for i in range(data['Kmax'])],
            'dim3_samples': sample_names, # np.char.encode(sample_names, 'ascii'),
            'dim1_drugs': self.FD.selected_drugs
            }

            for dim, labels in column_names.items():
                dset.attrs[dim] = labels


        for stat in ['ME', 'LOR', 'LRR']:
            if stat=='ME':
                postmean = ME.detach().numpy()
            elif stat=='LOR':
                postmean = LOR.detach().numpy()
            else:
                postmean = LRR.detach().numpy()
            for d in range(data['D']):
                postmean[d,:,:][~MASK] = float('nan')
            with h5py.File(os.path.join(dir_save,model_name+'_{0}.h5'.format(stat)), 'w') as f:
                # Create a dataset
                dset = f.create_dataset('{0}_posterior_mean'.format(stat), data=postmean)

                column_names = {
                'dim2_subclones': [i for i in range(data['Kmax'])],
                'dim3_samples': sample_names, # np.char.encode(sample_names, 'ascii'),
                'dim4_dimensions': self.feature_names,
                'dim1_drugs': self.FD.selected_drugs
                }

                for dim, labels in column_names.items():
                    dset.attrs[dim] = labels
                    
        return data_sample
    
    def get_log_likelihood(self, dic, params):
        """
        Compute the log likelihood of the observed data using the learned parameters.
        """
        data = self.merge_data_param(dic, params)
        N,Kmax,R,C,D = data['N'],data['Kmax'],data['R'],data['C'],data['D']
        R,D,Ndrug = data['n_r'].shape
        beta = data['beta']
        masks = data['masks']
                    
        pi = self.get_survival_probas(data)
            
        proportions = (data['proportions'] / torch.sum(data['proportions'], dim=1).unsqueeze(1)).T
        proportions = proportions[:,-N:]

        if self.mode_nu =="noise_correction":
            beta_control = data['beta_control']

        if self.mode_theta=='equal':
            theta_fd=data['theta_rna']
        else:
            theta_fd=data['theta_fd']

        theta_rna = data['theta_rna']

        
        log_likelihood = 0
        count = 0 # counting observations
        
        # --- First loop: DirichletMultinomial computations ---
        # Get nonzero indices for all `i` values efficiently
        idxs_notnull = proportions > 0  # Boolean mask for all `i`
        idxs_notnull = idxs_notnull.nonzero(as_tuple=True)  # Get valid indices

        # Compute log-likelihood using vectorized operations
        if data['n_rna'] is not None:
            if self.mode_theta in ['equal', 'shared', 'not shared decoupled']:
                log_likelihood += dist.DirichletMultinomial(
                    theta_rna * proportions[idxs_notnull],
                    torch.sum(dic['n_rna'][idxs_notnull], dim=0)
                ).log_prob(dic['n_rna'][idxs_notnull]).sum()  # Sum over valid indices
            elif self.mode_theta == 'not shared coupled':
                log_likelihood += dist.DirichletMultinomial(
                    theta_rna * theta_fd[idxs_notnull[1]] * proportions[idxs_notnull],
                    torch.sum(dic['n_rna'][idxs_notnull], dim=0)
                ).log_prob(dic['n_rna'][idxs_notnull]).sum()

        count += len(idxs_notnull[0])  # Update count

        # --- Second loop: BetaBinomial computations ---
        # Extract control mask indices efficiently
        control_mask_idxs = masks['C'].nonzero(as_tuple=True)  # Get valid `(c, i)`

        # Compute nu_healthy_c and nu_tumor_c efficiently
        if self.mode_nu == "fixed":
            nu_tumor_over_nu_healthy = torch.ones(1, device=proportions.device)
            nu_healthy_c = torch.ones(C, N, device=proportions.device) / 2  # Since 1 / (1+1) = 0.5
        else:
            nu_tumor_over_nu_healthy = torch.exp(data['X_nu_control'] @ beta_control)  # Vectorized
            nu_healthy_c = 1 / (1 + nu_tumor_over_nu_healthy)

        nu_tumor_c = 1 - nu_healthy_c  # Vectorized

        # Get healthy and tumor clusters
        healthy_clusters = self.cat2clusters['healthy']
        tumor_clusters = self.cat2clusters['tumor']

        # Compute terms for all (c, i) at once
        if healthy_clusters:
            healthy_term = (
                theta_fd.view(1, -1)  # Reshape for broadcasting
                * torch.sum(proportions[healthy_clusters, :], dim=0)[None,:]
                * nu_healthy_c
            )
        else:
            healthy_term = torch.zeros((C, N), device=proportions.device)

        if tumor_clusters:
            tumor_term = (
                theta_fd.view(1, -1)
                * torch.sum(proportions[tumor_clusters, :], dim=0)[None,:]
                * nu_tumor_c
            )
        else:
            tumor_term = torch.zeros((C, N), device=proportions.device)

        healthy_term = torch.clamp(healthy_term, min=self.EPSILON)
        tumor_term = torch.clamp(tumor_term, min=self.EPSILON)
        # Apply control mask
        log_likelihood += dist.BetaBinomial(
            healthy_term[control_mask_idxs],
            tumor_term[control_mask_idxs],
            total_count=dic['n_c'][control_mask_idxs]
        ).log_prob(dic['n0_c'][control_mask_idxs]).sum()

        count += len(control_mask_idxs[0])  # Update count


        ################### WELLS WITH DRUG
        # Precompute theta_fd_mode to avoid repetitive indexing
        theta_fd_mode = theta_fd[:N] if 'not' in self.mode_theta else theta_fd

        # Compute nu_healthy_drug and nu_tumor_drug efficiently
        if self.mode_nu == "fixed":
            nu_tumor_over_nu_healthy = torch.ones(1, device=proportions.device)
            nu_healthy_drug = torch.ones(R, D, N, device=proportions.device) / 2  # Since 1 / (1+1) = 0.5
        else:
            nu_tumor_over_nu_healthy = torch.exp(data['X_nu_drug'] @ beta_control)  # Vectorized
            nu_healthy_drug = 1 / (1 + nu_tumor_over_nu_healthy)

        nu_tumor_drug = 1 - nu_healthy_drug  # Vectorized

        # Get healthy and tumor clusters
        healthy_clusters = self.cat2clusters['healthy']
        tumor_clusters = self.cat2clusters['tumor']

        # Compute healthy and tumor terms only if clusters exist
        if healthy_clusters:
            healthy_term = (
                theta_fd_mode.view(1, 1, -1)  # Reshape to broadcast properly
                * torch.sum(proportions[None,healthy_clusters, :] * pi[:, healthy_clusters, :], dim=1)[None,:,:]
                * nu_healthy_drug
            )
        else:
            healthy_term = torch.zeros((R, D, N), device=proportions.device)

        if tumor_clusters:
            tumor_term = (
                theta_fd_mode.view(1, 1, -1)
                * torch.sum(proportions[None, tumor_clusters, :] * pi[:, tumor_clusters, :], dim=1)[None,:,:]
                * nu_tumor_drug
            )
        else:
            tumor_term = torch.zeros((R, D, N), device=proportions.device)

        healthy_term = torch.clamp(healthy_term, min=self.EPSILON)
        tumor_term = torch.clamp(tumor_term, min=self.EPSILON)
        # Apply mask efficiently (removes unnecessary iterations)
        masked_indices = masks['R'].nonzero(as_tuple=True)  # Get valid (r,d,i) indices
        log_likelihood += dist.BetaBinomial(
            healthy_term[masked_indices],
            tumor_term[masked_indices],
            total_count=dic['n_r'][masked_indices]
        ).log_prob(dic['n0_r'][masked_indices]).sum()  # Sum for efficiency

        # Count the number of valid cases
        count += len(masked_indices[0])
        return (log_likelihood/count)
    

    def get_log_likelihood_slow(self, dic, params):
        """
        Compute the log likelihood of the observed data using the learned parameters.
        
        .. deprecated::
            This method is retained for validation purposes only. 
            Use :meth:`get_log_likelihood` for better performance.
        """
        import warnings
        warnings.warn(
            "get_log_likelihood_slow is deprecated and significantly slower. "
            "Use get_log_likelihood instead.",
            DeprecationWarning,
            stacklevel=2
        )
        data = self.merge_data_param(dic, params)
        N,Kmax,R,C,D = data['N'],data['Kmax'],data['R'],data['C'],data['D']
        R,D,Ndrug = data['n_r'].shape
        beta = data['beta']
        masks = data['masks']
                    
        pi = self.get_survival_probas(data)
            
        proportions = (data['proportions'] / torch.sum(data['proportions'], dim=1).unsqueeze(1)).T
        proportions = proportions[:,-N:]

        if self.mode_nu =="noise_correction":
            beta_control = data['beta_control']

        if self.mode_theta=='equal':
            theta_fd=data['theta_rna']
        else:
            theta_fd=data['theta_fd']

        theta_rna = data['theta_rna']

        
        log_likelihood = 0
        count = 0 # counting observations
        
        for i in range(N):
            idxs_notnull = np.where(proportions[:, i] > 0)[0]
            if not (data['n_rna'] is None):  # for the bulk model
                if self.mode_theta in ['equal', 'shared', 'not shared decoupled']:
                    log_likelihood += dist.DirichletMultinomial(
                        theta_rna * proportions[idxs_notnull, i],
                        torch.sum(dic['n_rna'][idxs_notnull, i])
                    ).log_prob(dic['n_rna'][idxs_notnull, i])
                elif self.mode_theta == 'not shared coupled':
                    log_likelihood += dist.DirichletMultinomial(
                        theta_rna * theta_fd[i] * proportions[idxs_notnull, i],
                        torch.sum(dic['n_rna'][idxs_notnull, i])
                    ).log_prob(dic['n_rna'][idxs_notnull, i])
                count += 1

        for i in range(N):
            # CONTROL WELLS
            for c in range(C):
                if masks['C'][c, i]:
                    if self.mode_nu == "fixed":
                        nu_tumor_over_nu_healthy = torch.tensor(1)
                        nu_healthy_c = 1. / (1 + nu_tumor_over_nu_healthy)
                    elif self.mode_nu == "noise_correction":
                        nu_tumor_over_nu_healthy = torch.exp(data['X_nu_control'] @ beta_control)
                        nu_healthy_c = 1. / (1 + nu_tumor_over_nu_healthy)

                    nu_tumor_c = 1 - nu_healthy_c
                
                    healthy_clusters = self.cat2clusters['healthy']
                    tumor_clusters = self.cat2clusters['tumor']

                    # Handle empty cluster cases
                    if len(healthy_clusters) > 0:
                        healthy_term = theta_fd[i] * torch.sum(proportions[healthy_clusters, i]) * nu_healthy_c[c, i]
                    else:
                        healthy_term = torch.tensor(0.0, device=proportions.device)

                    if len(tumor_clusters) > 0:
                        tumor_term = theta_fd[i] * torch.sum(proportions[tumor_clusters, i], dim=0) * (1 - nu_healthy_c[c, i])
                    else:
                        tumor_term = torch.tensor(0.0, device=proportions.device)

                    log_likelihood += dist.BetaBinomial(
                        healthy_term, tumor_term,  # Ensuring correct tensor shape
                        total_count=dic['n_c'][c, i]
                    ).log_prob(dic['n0_c'][c, i])

                    #log_likelihood += dist.BetaBinomial(theta_fd[i]*proportions[self.cat2clusters['healthy'],i]*nu_healthy_c[c,i], theta_fd[i]*torch.sum(proportions[self.cat2clusters['tumor'],i], dim=0)* (1 - nu_healthy_c[c,i]), total_count=dic['n_c'][c,i]).log_prob(dic['n0_c'][c,i])
                    count += 1

        for i in range(Ndrug):
            for d in range(D):
                for r in range(R):
                    if masks['R'][r, d, i]:
                        if self.mode_nu == "fixed":
                            nu_tumor_over_nu_healthy = torch.tensor(1)
                            nu_healthy_drug = 1. / (1 + nu_tumor_over_nu_healthy)
                        elif self.mode_nu == "noise_correction":
                            nu_tumor_over_nu_healthy = torch.exp(data['X_nu_drug'] @ beta_control)
                            nu_healthy_drug = 1. / (1 + nu_tumor_over_nu_healthy)
                        nu_tumor_drug = 1 - nu_healthy_drug
                        if 'not' in self.mode_theta:
                            theta_fd_mode = theta_fd[:Ndrug]
                        else:
                            theta_fd_mode = theta_fd
                            
                        healthy_clusters = self.cat2clusters['healthy']
                        tumor_clusters = self.cat2clusters['tumor']

                        # Handle empty cluster cases
                        if len(healthy_clusters) > 0:
                            healthy_term = theta_fd_mode[i] * torch.sum(proportions[healthy_clusters, i] * pi[d, healthy_clusters, i]) * nu_healthy_drug[r, d, i]
                        else:
                            healthy_term = torch.tensor(0.0, device=proportions.device)

                        if len(tumor_clusters) > 0:
                            tumor_term = theta_fd_mode[i] * torch.sum(proportions[tumor_clusters, i] * pi[d, tumor_clusters, i]) * (1 - nu_healthy_drug[r, d, i])
                        else:
                            tumor_term = torch.tensor(0.0, device=proportions.device)

                        log_likelihood += dist.BetaBinomial(
                            healthy_term, tumor_term,  # Ensure this has the right shape
                            total_count=dic['n_r'][r, d, i]
                        ).log_prob(dic['n0_r'][r, d, i])

                        # log_likelihood += dist.BetaBinomial(theta_fd_mode[i]*proportions[self.cat2clusters['healthy'],i]*pi[d,self.cat2clusters['healthy'],i]*nu_healthy_drug[r,d,i], theta_fd_mode[i]*torch.sum(proportions[self.cat2clusters['tumor'],i]*pi_tumor[d,self.cat2clusters['tumor'],i])* (1 - nu_healthy_drug[r,d,i]), total_count=dic['n_r'][r,d,i]).log_prob(dic['n0_r'][r,d,i])
                        count += 1
        return log_likelihood / count
    
    
    def sampling_from_posterior_with_posterior_stds(self, data, dir_save, params=None, nb_ites=100, sample_names=None):
        """
        Use the generative model to sample new data using the learned parameters.
        This method is only useful when features are defined at the cell level. We sample from the posterior distribution the parameters 'gamma_{clonelabel}' for all clonelabels in order to compute the posterior mean and stds of the different statistics of interest.
        """
        if sample_names is None:
            sample_names = [i for i in range(data['N'])]
        ### Dealing with the posterior
        N, D, R, Kmax = data['N'], data['D'], data['R'], data['Kmax']
        
        ## data_sample will be used to store the posterior means of the different model parameters.
        data_sample = deepcopy(data)
        for key, val in params.items():
            data_sample[key] = torch.tensor(val)
        data_sample['proportions'] = data_sample['proportions']
        if self.mode_theta != "equal":
            data_sample['theta_fd'] = data_sample['theta_fd']
        
        ## data_esti_ghost contains the observed data and the learned model parameters. It will be used to sample from the posterior.
        data_esti_ghost = deepcopy(data)
        for key, val in params.items():
            data_esti_ghost[key] = torch.tensor(val)
        data_esti_ghost['proportions'] = data_esti_ghost['proportions']
        if self.mode_theta != "equal":
            data_esti_ghost['theta_fd'] = data_esti_ghost['theta_fd']
            
            
        try:
            guide = self.guide
            def sample_latent_variables():
                return guide.sample_latent()
        except:
            guide = self.get_optimized_guide(params)
            def sample_latent_variables():
                return guide.sample()
            
        ## We initialize the variables of interest to zeros in data_sample
        samp_gamma = sample_latent_variables()
        dim = len(samp_gamma)//self.n_clonelabels
        for i, clonelabel in enumerate(self.clonelabels):
            data_esti_ghost['gamma_{0}'.format(clonelabel)] = samp_gamma[dim*i:dim*(i+1)]
        data_sample_ghost = self.sampling(data_esti_ghost)
        for key,val in data_sample_ghost.items():
            data_sample[key] = val
        ls_var_2_avg = ['n0_c', 'n0_r',  'n_rna', 'frac_r', 'frac_c', 'frac_mean_r', 'frac_mean_c', 'log_score', 'pi','nu_healthy_drug', 'nu_healthy_control']
        for var in ls_var_2_avg:
            data_sample[var] = torch.zeros(data_sample_ghost[var].shape)


        PI = torch.zeros((nb_ites,D,Kmax,N))
        latent_dim = params['beta'].shape[1]
        PI = torch.zeros((nb_ites,D,Kmax,N))
        LRR = torch.zeros((nb_ites,D,Kmax,N,latent_dim))
        LOR = torch.zeros((nb_ites,D,Kmax,N,latent_dim))
        ME = torch.zeros((nb_ites,D,Kmax,N,latent_dim))
        subclone_features = torch.zeros((nb_ites,Kmax,N,latent_dim))
        for i in tqdm(range(nb_ites)):
            samp_gamma = sample_latent_variables()
            for j, clonelabel in enumerate(self.clonelabels):
                data_esti_ghost['gamma_{0}'.format(clonelabel)] = samp_gamma[dim*j:dim*(j+1)]
            data_sample_ghost = self.sampling(data_esti_ghost)
            
            # params_pi = {"beta":data['beta']}
            # for clonelabel in self.clonelabels:
            #     params_pi["gamma_{0}".format(clonelabel)] = data_esti_ghost["gamma_{0}".format(clonelabel)]
            #     params_pi["offset_{0}".format(clonelabel)] = data["offset_{0}".format(clonelabel)]
            pi = self.compute_survival_probas_single_cell_features(data, data_esti_ghost)
            PI[i,:,:,:] = pi.detach()
            
            for clonelabel in self.clonelabels:
                idxs = self.clonelabel2clusters[clonelabel]
                subclone_features[i,idxs,:,:] = torch.sum(data['X'][idxs,:,:,:] * (masked_softmax(torch.matmul(data['X'][idxs,:,:,:], data_esti_ghost['gamma_{0}'.format(clonelabel)]), data['masks']['SingleCell'][idxs,:,:], dim=2)).unsqueeze(3), dim=2)


            for var in ls_var_2_avg:
                data_sample[var] += data_sample_ghost[var]/nb_ites
            ## END: Dealing with posterior

            for j in range(latent_dim):
                params_pi = {}
                for l, clonelabel in enumerate(self.clonelabels):
                    params_pi["gamma_{0}".format(clonelabel)] = samp_gamma[dim*l:dim*(l+1)]
                    params_pi["offset_{0}".format(clonelabel)] = torch.tensor(params["offset_{0}".format(clonelabel)])
                beta_j = deepcopy(params['beta'])                
                beta_j[:,j] = 0
                params_pi["beta"] = torch.tensor(beta_j)
                pi_j = self.compute_survival_probas_single_cell_features(data, params_pi)

                # LOR
                LOR[i,:,:,:,j] = torch.log(pi/(1-pi))  -  torch.log(pi_j/(1-pi_j))

                # LRR
                LRR[i,:,:,:,j] = torch.log(pi/pi_j)

            # ME: ite x D x Kmax x N x dim
            ME[i,:,:,:,:] = (data_sample_ghost['pi'] * (1-data_sample_ghost['pi']))[:,:,:,None] * params['beta'][:,None,None,:]

            
        MASK = data['masks']['RNA']
        if np.sum(MASK.detach().numpy())==0:
            MASK = torch.sum(data['masks']['SingleCell'], dim=2)>0.5
            

        subclone_features = subclone_features.detach().numpy()
        all_local_importances = params['beta'][None,:,None,None,:] * subclone_features[:,None,:,:,:]
        with h5py.File(os.path.join(dir_save,'local_importance.h5'), 'w') as f:
            # Create a dataset
            dset = f.create_dataset('local_importance_mean', data=np.mean(all_local_importances, axis=0))

            column_names = {
            'dim2_subclones': [i for i in range(data['Kmax'])],
            'dim3_samples': sample_names, #np.char.encode(sample_names_test, 'ascii'),
            'dim4_dimensions': self.feature_names,
            'dim1_drugs':  self.FD.selected_drugs
            }

            for dim, labels in column_names.items():
                dset.attrs[dim] = labels

            dset = f.create_dataset('local_importance_std', data=np.std(all_local_importances, axis=0))

            column_names = {
            'dim2_subclones': [i for i in range(Kmax)],
            'dim3_samples': sample_names,
            'dim4_dimensions': self.feature_names,
            'dim1_drugs':  self.FD.selected_drugs
            }

            for dim, labels in column_names.items():
                dset.attrs[dim] = labels

        postmean = np.mean(subclone_features, axis=0)
        poststd = np.std(subclone_features, axis=0)
        for j in range(latent_dim):
            postmean[:,:,j][~MASK] = float('nan')
            poststd[:,:,j][~MASK] = float('nan')
        with h5py.File(os.path.join(dir_save,'subclone_features.h5'), 'w') as f:
            # Create a dataset
            dset = f.create_dataset('subclone_features_posterior_mean', data=postmean)
            column_names = {
            'dim1_subclones': [i for i in range(data['Kmax'])],
            'dim2_samples': sample_names, # np.char.encode(sample_names, 'ascii'),
            'dim3_dimensions': self.feature_names
            }

            for dim, labels in column_names.items():
                dset.attrs[dim] = labels

            dset = f.create_dataset('subclone_features_posterior_std', data=poststd)

            column_names = {
            'dim1_subclones': [i for i in range(Kmax)],
            'dim2_samples': sample_names, # np.char.encode(sample_names, 'ascii'),
            'dim3_dimensions': self.feature_names
            }

            for dim, labels in column_names.items():
                dset.attrs[dim] = labels

        postmean = torch.mean(PI, dim=0).numpy()
        poststd = torch.std(PI, dim=0).numpy()
        for d in range(data['D']):
            postmean[d,:,:][~MASK] = float('nan')
            poststd[d,:,:][~MASK] = float('nan')
        with h5py.File(os.path.join(dir_save,'survival_probabilities.h5'), 'w') as f:
            # Create a dataset
            dset = f.create_dataset('survival_probabilities_posterior_mean', data=postmean)
            column_names = {
            'dim2_subclones': [i for i in range(data['Kmax'])],
            'dim3_samples': sample_names, # np.char.encode(sample_names, 'ascii'),
            'dim1_drugs': self.FD.selected_drugs
            }

            for dim, labels in column_names.items():
                dset.attrs[dim] = labels

            dset = f.create_dataset('survival_probabilities_posterior_std', data=torch.std(PI, dim=0).numpy())

            column_names = {
            'dim2_subclones': [i for i in range(data['Kmax'])],
            'dim3_samples': sample_names, #  np.char.encode(sample_names, 'ascii'),
            'dim1_drugs': self.FD.selected_drugs
            }

            for dim, labels in column_names.items():
                dset.attrs[dim] = labels

        for stat in ['ME', 'LOR', 'LRR']:
            if stat=='ME':
                postmean = torch.mean(ME, dim=0).detach().numpy()
                poststd = torch.std(ME, dim=0).detach().numpy()
            elif stat=='LOR':
                postmean = torch.mean(LOR, dim=0).detach().numpy()
                poststd = torch.std(LOR, dim=0).detach().numpy()
            else:
                postmean = torch.mean(LRR, dim=0).detach().numpy()
                poststd = torch.std(LRR, dim=0).detach().numpy()
            for d in range(data['D']):
                postmean[d,:,:][~MASK] = float('nan')
                poststd[d,:,:][~MASK] = float('nan')
            with h5py.File(os.path.join(dir_save,'_{0}.h5'.format(stat)), 'w') as f:
                # Create a dataset
                dset = f.create_dataset('{0}_posterior_mean'.format(stat), data=postmean)

                column_names = {
                'dim2_subclones': [i for i in range(data['Kmax'])],
                'dim3_samples': sample_names, # np.char.encode(sample_names, 'ascii'),
                'dim4_dimensions': self.feature_names,
                'dim1_drugs': self.FD.selected_drugs
                }

                for dim, labels in column_names.items():
                    dset.attrs[dim] = labels

                dset = f.create_dataset('{0}_posterior_std'.format(stat), data=poststd)

                column_names = {
                'dim2_subclones': [i for i in range(data['Kmax'])],
                'dim3_samples': sample_names, # np.char.encode(sample_names, 'ascii'),
                'dim4_dimensions': self.feature_names,
                'dim1_drugs': self.FD.selected_drugs
                }

                for dim, labels in column_names.items():
                    dset.attrs[dim] = labels
                    
        return data_sample
    
