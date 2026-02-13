
import numpy as np
import torch
import pyro
from torch.distributions import constraints
from pyro import poutine
import pyro.distributions as dist
import torch.distributions as tdist
from pyro.infer.autoguide import (
    AutoMultivariateNormal,
    init_to_mean,
    AutoLowRankMultivariateNormal,
    AutoDiagonalNormal,
    AutoDelta
)
from pyro.infer import SVI, Trace_ELBO


def L2_regularizer(my_parameters: torch.Tensor) -> torch.Tensor:
    """Compute L2 regularization term.
    
    Parameters
    ----------
    my_parameters : torch.Tensor
        Parameters to regularize
        
    Returns
    -------
    torch.Tensor
        L2 regularization loss
    """
    reg_loss = 0.0
    for param in my_parameters:
        reg_loss = reg_loss + param.pow(2.0).sum()
    return reg_loss / np.prod(my_parameters.shape)


class Trainer:
    """Class used to train the model using SVI or SGD."""

    def __init__(self, type_guide: str = "full_MVN", rank: int = None):
        """Initialize the trainer.
        
        Parameters
        ----------
        type_guide : str, default="full_MVN"
            Type of guide distribution: "full_MVN", "lowrank_MVN", or "diagonal"
        rank : int, optional
            Rank for low-rank MVN guide (default: 20)
        """
        self.type_guide = type_guide
        if type_guide == "lowrank_MVN":
            if rank is None:
                self.rank = 20
            else:
                self.rank = rank

    def svi(
        self,
        model,
        guide,
        data_train_svi: dict,
        ls_params_obs: list,
        penalty_l2: float = None,
        penalty_l1: float = None,
        lr: float = 0.001,
        n_steps: int = 1000
    ) -> None:
        """Run stochastic variational inference.
        
        Parameters
        ----------
        model : callable
            Model function
        guide : callable
            Guide function
        data_train_svi : dict
            Training data dictionary
        ls_params_obs : list
            List of observed parameter names
        penalty_l2 : float, optional
            L2 penalty coefficient
        penalty_l1 : float, optional
            L1 penalty coefficient
        lr : float, default=0.001
            Learning rate
        n_steps : int, default=1000
            Number of training steps
        """
        pyro.clear_param_store()
        normalize = np.sum([
            torch.sum(data_train_svi['masks'][el]) for el in ['C', 'R', 'RNA']
        ])
        
        def loss_fn(model, guide):
            return pyro.infer.Trace_ELBO().differentiable_loss(model, guide, data_train_svi)
        
        with pyro.poutine.trace(param_only=True) as param_capture:
            loss = loss_fn(model, guide)
        params = set(
            site["value"].unconstrained() for site in param_capture.trace.nodes.values()
        )
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.90, 0.999))
        
        for step in range(n_steps):
            # Compute loss
            loss = loss_fn(model, guide) / normalize

            if penalty_l1 is not None:
                shapebeta = pyro.param('beta').shape
                loss += penalty_l1 * torch.nn.L1Loss()(
                    pyro.param('beta'), torch.zeros(shapebeta)
                )
            
            if penalty_l2 is not None:
                if self.mode_nu == "fixed":
                    loss += penalty_l2 * L2_regularizer(pyro.param('beta'))
                elif self.mode_nu == "noise_correction":
                    loss += penalty_l2 * L2_regularizer(pyro.param('beta'))
                    loss += torch.matmul(
                        torch.matmul(pyro.param('beta_control'), self.Ssplines),
                        pyro.param('beta_control')
                    )

            loss.backward()
            # Take a step and zero the parameter gradients
            optimizer.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(f'[iter {step}]  loss: {loss:.4f}')



    def train_single_cell_features(self, data_train, penalty_l1 = None, penalty_l2 = None, lr=0.01, n_steps=2000):     
        ls_params_obs = ['n0_c', 'n_c', 'n0_r', 'n_r', 'n_rna', 'masks', 'X', 'frac_r', 'frac_c']
        data_train_svi = {param: data_train[param] for param in ls_params_obs}
        ls_params_obs = ['n0_c', 'n0_r', 'n_rna', 'masks', 'X', 'X_nu_control', 'X_nu_drug']

        pyro.clear_param_store()

        self.svi(lambda x: self.prior(x, fixed_proportions=False), self.guide, data_train, ls_params_obs, penalty_l1 = penalty_l1, penalty_l2 = penalty_l2, lr=lr, n_steps=n_steps)

        # Saving the learned parameters
        params_svi = {}
        for key,val in pyro.get_param_store().named_parameters():
            params_svi[key] = pyro.param(key).detach().numpy()
        return params_svi
    
    def train_subclone_features(self, data_train_svi, guide, penalty_l2=None, penalty_l1=None, lr=0.01, n_steps=1000):
        pyro.clear_param_store()
        normalize = np.sum([
            torch.sum(data_train_svi['masks'][el]) for el in ['C', 'R', 'RNA']
        ])

        def loss_fn(model, guide):
            return pyro.infer.Trace_ELBO().differentiable_loss(
                model, guide, data_train_svi
            )
        
        with pyro.poutine.trace(param_only=True) as param_capture:
            loss = loss_fn(self.prior, guide)
        params = set(
            site["value"].unconstrained() for site in param_capture.trace.nodes.values()
        )
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.90, 0.999))
        
        for step in range(n_steps):
            # compute loss
            loss = loss_fn(self.prior, guide)/normalize
            if not(penalty_l1 is None):
                shapebeta = pyro.param('beta').shape
                loss += penalty_l1 * torch.nn.L1Loss()(pyro.param('beta'), torch.zeros(shapebeta) )
            if not(penalty_l2 is None):
                loss += penalty_l2 * (L2_regularizer(pyro.param('beta')))
                    
            loss.backward()
            # take a step and zero the parameter gradients
            optimizer.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print('[iter {}]  loss: {:.4f}'.format(step, loss))
                
        # Saving the learned parameters
        params_svi = {}
        for key,val in pyro.get_param_store().named_parameters():
            params_svi[key] = pyro.param(key).detach().numpy()
        return params_svi
                
    def train(self, data_train_svi, penalty_l1 = None, penalty_l2 = None, lr=0.01, n_steps=2000):
        """
        Trains the model using stochastic variational inference if features are defined at the single cell level. In case the features are defined at the subclone level, stochastic gradient descent is used to get the MLE estimates of the model parameters.
        Parameters
        ----------
        data_train_svi: dictionary
            dictionary containing the training data.
        penalty_l1: float
            if not None, hyperparameter for the L1 penalty on the beta's coefficients (representing the learned drug features).
        penalty_l2: float
            if not None, hyperparameter for the L2 penalty on the beta's coefficients (representing the learned drug features).
        lr: float
            learning rate.
        n_steps: int
            total number of iterations of the algorithm (SVI or SGD). 
        """
        if data_train_svi['single_cell_features']:
            if self.type_guide=="full_MVN":
                self.guide = AutoMultivariateNormal(self.prior, init_loc_fn=init_to_mean)
            elif self.type_guide=="lowrank_MVN":
                self.guide = AutoLowRankMultivariateNormal(self.prior, rank=self.rank)  # Reduce rank
            elif self.type_guide=="diagonal":
                self.guide = AutoDiagonalNormal(self.prior)
            return self.train_single_cell_features(data_train_svi, penalty_l2=penalty_l2, penalty_l1=penalty_l1, lr=lr, n_steps=n_steps)

        else:
            def guide(dic):
                pass
            return self.train_subclone_features(data_train_svi, guide, penalty_l2=penalty_l2, penalty_l1=penalty_l1, lr=lr, n_steps=n_steps)
