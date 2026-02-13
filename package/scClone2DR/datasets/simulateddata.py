import numpy as np
import torch
import pyro
from torch.distributions import constraints
from pyro import poutine
import pyro.distributions as dist
import torch.distributions as tdist
from ..utils import *
import pickle
import copy
from .basedataset import BaseDataset

class SimulatedData(BaseDataset):

    def __init__(self):
        super(SimulatedData, self).__init__()
        
        
    def get_simulated_training_data(self, data_train=None, neg_bin_n=2, theta_rna_per_sample=False, mode_nu="noise_correction", mode_theta='not shared decoupled'):
        
        self.mode_nu = mode_nu
        self.mode_theta = mode_theta

        ############## YOU CAN CHOOSE THE DESIRED SETTING
        if data_train is None:
            SETTING = 'EASY'
            settings = {'HARD':{'disp':20., 'etheta':3.}, 'EASY':{'disp':100., 'etheta':100.}}

            C,R,Ntrain,Kmax = 24,10,30,7
            data_train = {'C':C,'R':R,'N':Ntrain,'D':D, 'Kmax':Kmax, 'single_cell_features':False}

            # The two parameters controlling the overdispersion
            data_train['dispersion_fd'] = settings[SETTING]['disp']
            data_train['etheta_fd'] = settings[SETTING]['etheta']
            var_preassay = 0.03
        else:
            Ntrain,Kmax,R,C,D = data_train['N'],data_train['Kmax'],data_train['R'],data_train['C'],data_train['D']
            data_train['single_cell_features'] = False
            try:
                var_preassay = data_train['var_preassay']
            except:
                var_preassay = 0.03

        self.cluster2clonelabel = ['healthy'] + ['tumor' for i in range(Kmax-1)]
        self.clonelabel2cat = {'healthy':'healthy', 'tumor':'tumor'}
        self.init_cat_clonelabel()

        sample2subclone_len = {}

        # Definition of the masks (in case some sample )
        masks_train = {}
        masks_train['RNA'] = torch.ones((Kmax,Ntrain), dtype=torch.bool)
        masks_train['C'] = torch.ones((C,Ntrain), dtype=torch.bool)
        masks_train['R'] = torch.ones((R,D,Ntrain), dtype=torch.bool)
        for i in range(Ntrain):
            sample2subclone_len[i] = [j for j in range(i*Kmax,(i+1)*Kmax)]
        data_train['masks'] = masks_train
        pyro.clear_param_store()

    #         data_train['nu_healthy'] = torch.tensor(0.8)
    #         data_train['nu_tumor'] = 1-data_train['nu_healthy']

        props = torch.zeros((Kmax,Ntrain))
        for i in range(Ntrain):
            lensub = len(sample2subclone_len[i])
            props[:lensub,i] = torch.distributions.Dirichlet(torch.tensor([4.]+(lensub-1)*[1.])).sample()

        data_train['proportions'] = props.T

        # Design matrix (columns of 'X' are features)
        dim = 20
        dim_all = 20
        self.feature_names = ["dim_{0}".format(i) for i in range(dim_all)]
        data_train['X'] = torch.tensor(np.abs(np.random.normal(0,0.3,(Kmax,Ntrain,dim_all)))).float()
        for k in range(Kmax):
            data_train['X'][k,:,:] *= (1- k/(Kmax))
        data_train['beta'] = torch.zeros((D,dim_all))
        for d in range(D):
            data_train['beta'][d,:dim] = torch.tensor(np.abs(np.random.normal(0,1,dim))).float() / np.sqrt(dim)

        data_train['offset_healthy'] = torch.zeros(D)

        if neg_bin_n>=1:
            theta_fd = np.random.negative_binomial(neg_bin_n, 0.001, Ntrain)
        else:
            theta_fd = np.array(1000.*neg_bin_n*np.ones(Ntrain), dtype=np.float)
    #np.concatenate([np.random.negative_binomial(3*i, 0.01, int(Ntrain//6)+1) for i in [4,10,20,30,40,50]])
        data_train['theta_fd'] = torch.tensor(theta_fd[:Ntrain]).float()
        #torch.tensor(np.random.normal(2000, 100, Ntrain))#500 * torch.ones(Ntrain)
        data_train['offset_tumor'] = torch.zeros(D)
        data_train['beta_control'] = torch.ones(1).float()
        data_train['X_nu_control'] = torch.tensor(np.random.normal(np.log(0.6/0.9),var_preassay,(C,Ntrain,1))).float() #torch.tensor(np.abs(np.random.normal(0,0.3,(C,Ntrain,1)))).float()
        data_train['X_nu_drug'] = torch.tensor(np.random.normal(np.log(0.6/0.9),var_preassay,(R,D,Ntrain,1))).float()  #torch.tensor(np.abs(np.random.normal(0,0.3,(R,D,Ntrain,1)))).float()

        # Average number of cells in RNA data (for sampling)
        data_train['n_rna'] = (5000*torch.ones((data_train['Kmax'],data_train['N']))).int()
        # Average number of cells in wells (for sampling)
        data_train['n_r'] = (1000* torch.ones(R,D,data_train['N'])).int()
        data_train['n_c'] = (1000* torch.ones(C,data_train['N']) ).int()

        pyro.clear_param_store()
        ####### We sample the training data
        data_samp = self.sampling(data_train)

        data_train = load_from_sampling(data_train,data_samp)

        return data_train

    def get_data_split_simu(self, data, idxs_train, idxs_test):
        Ntrain = len(idxs_train)
        Ntest = len(idxs_test)
        Ntot = Ntrain + Ntest
        masks_train = {}
        Kmax,R,C,D = data['Kmax'],data['R'],data['C'],data['D']
        masks_train['RNA'] = torch.ones((Kmax,Ntot), dtype=torch.bool)
        for i in range(Ntrain):
            for k in range(Kmax):
                masks_train['RNA'][k,i] = data['masks']['RNA'][k,idxs_train[i]]
        for i in range(Ntest):
            for k in range(Kmax):
                masks_train['RNA'][k,Ntrain+i] = data['masks']['RNA'][k,idxs_test[i]]

        masks_train['C'] = torch.ones((C,Ntot), dtype=torch.bool)
        for i in range(Ntrain):
            for c in range(data['C']):
                masks_train['C'][c,i] = data['masks']['C'][c,idxs_train[i]]
        for i in range(Ntest):
            for c in range(data['C']):
                masks_train['C'][c,Ntrain+i]  = data['masks']['C'][c,idxs_test[i]]

        masks_train['R'] = torch.ones((R,D,Ntrain), dtype=torch.bool)
        for idxdrug in range(D):
            for i in range(Ntrain):
                for r in range(data['R']):
                    masks_train['R'][r,idxdrug,i] = data['masks']['R'][r,idxdrug,idxs_train[i]]



        data_train = {'X': data['X'][:,idxs_train,:], 'D':D, 'R':R, 'C':C, 'Kmax':Kmax, 'N':Ntot, 'single_cell_features':False}
        data_train['X_nu_drug'] = data['X_nu_drug'][:,:,idxs_train,:]
        data_train['X_nu_control'] = torch.zeros(data['X_nu_control'].shape)
        data_train['X_nu_control'][:,:Ntrain,:] = data['X_nu_control'][:,idxs_train,:]
        data_train['X_nu_control'][:,Ntrain:,:] = data['X_nu_control'][:,idxs_test,:]

        data_train['n0_c'] = torch.zeros((C,Ntot))
        data_train['n0_c'][:,:Ntrain] = data['n0_c'][:,idxs_train]
        data_train['n0_c'][:,Ntrain:] = data['n0_c'][:,idxs_test]

        data_train['n_c'] = torch.zeros((C,Ntot))
        data_train['n_c'][:,:Ntrain] = data['n_c'][:,idxs_train]
        data_train['n_c'][:,Ntrain:] = data['n_c'][:,idxs_test]

        data_train['n_rna'] = torch.zeros((Kmax,Ntot))
        data_train['n_rna'][:,:Ntrain] = torch.tensor(data['n_rna'][:,idxs_train])
        data_train['n_rna'][:,Ntrain:] = torch.tensor(data['n_rna'][:,idxs_test])

        data_train['n0_r'] = data['n0_r'][:,:,idxs_train]
        data_train['n_r']  = data['n_r'][:,:,idxs_train]
        data_train['masks'] = masks_train
        data_train['proportions'] = torch.zeros((Ntot,Kmax)) 
        data_train['proportions'][:Ntrain,:] = data['proportions'][idxs_train,:]
        data_train['proportions'][Ntrain:,:] = data['proportions'][idxs_test,:]

        data_train['ini_proportions'] = (data_train['n_rna'] / torch.tile(torch.sum(data_train['n_rna'], dim=0).reshape(1,Ntot), (Kmax,1))).T

        frac_r = 1. - data_train['n0_r']/data_train['n_r']
        frac_c = 1. - data_train['n0_c']/data_train['n_c']
        data_train['frac_r'] = torch.nan_to_num(frac_r) 
        data_train['frac_c'] = torch.nan_to_num(frac_c) 

        frac_mean_r = torch.sum(masks_train['R']*torch.nan_to_num(frac_r), dim=0) / torch.sum(masks_train['R'], dim=0) 
        frac_mean_c = torch.sum(masks_train['C']*torch.nan_to_num(frac_c), dim=0) / torch.sum(masks_train['C'], dim=0) 


        R, D, N = data['n_r'].shape
        C, N = data['n_c'].shape
        Kmax, N = data['n_rna'].shape
        data_test = {'R':R, 'N':Ntest, 'D': D, 'C':C, 'Kmax':Kmax, 'single_cell_features':False}
        data_test['X'] = data['X'][:,idxs_test,:]
        data_test['X_nu_drug'] = data['X_nu_drug'][:,:,idxs_test,:]
        data_test['X_nu_control'] = data['X_nu_control'][:,idxs_test,:]

        data_test['n_rna'] = torch.tensor(data['n_rna'][:,idxs_test])
        data_test['n0_c'] = data['n0_c'][:,idxs_test]
        data_test['n_c']  = data['n_c'][:,idxs_test]
        data_test['n0_r'] = data['n0_r'][:,:,idxs_test]
        data_test['n_r']  = data['n_r'][:,:,idxs_test]
        data_test['proportions'] = data['proportions'][idxs_test,:]
        data_test['ini_proportions'] = (data_test['n_rna'] / torch.tile(torch.sum(data_test['n_rna'], dim=0).reshape(1,Ntest), (Kmax,1))).T


        masks_test = {}
        masks_test['RNA'] = torch.ones((Kmax,Ntest), dtype=torch.bool)
        for i in range(Ntest):
            for k in range(Kmax):
                masks_test['RNA'][k,i] = data['masks']['RNA'][k,idxs_test[i]]
        masks_test['C'] = torch.ones((C,Ntest), dtype=torch.bool)
        for i in range(Ntest):
            for c in range(data['C']):
                masks_test['C'][c,i] = data['masks']['C'][c,idxs_test[i]]
        masks_test['R'] = torch.ones((R,D,Ntest), dtype=torch.bool)
        for idxdrug in range(D):
            for i in range(Ntest):
                for r in range(R):
                    masks_test['R'][r,idxdrug,i] = data['masks']['R'][r,idxdrug,idxs_test[i]]

        data_test['masks'] = masks_test

        frac_r = 1. - data_test['n0_r']/data_test['n_r']
        frac_c = 1. - data_test['n0_c']/data_test['n_c']
        data_test['frac_r'] = torch.nan_to_num(frac_r) 
        data_test['frac_c'] = torch.nan_to_num(frac_c) 

        frac_mean_r = torch.sum(masks_test['R']*torch.nan_to_num(frac_r), dim=0) / torch.sum(masks_test['R'], dim=0) 
        frac_mean_c = torch.sum(masks_test['C']*torch.nan_to_num(frac_c), dim=0) / torch.sum(masks_test['C'], dim=0) 


        data_test['log_scores'] = torch.log(frac_mean_c.unsqueeze(0) / frac_mean_r)


        data_train['simulated_data'] = True
        data_test['simulated_data'] = True
        return data_train, data_test
    
    def save_data(self, data, path, name_dataset):
        with open(path+name_dataset+'pkl', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def get_base_from_data(self, dic):
        data = copy.deepcopy(dic)
        data['X'] = torch.zeros(data['X'].shape)
        self.cluster2clonelabel = ['healthy'] + ['tumor' for i in range(data['Kmax']-1)]
        self.clonelabel2cat = {'healthy':'healthy', 'tumor':'tumor'}
        self.init_cat_clonelabel()
        return data
    
    
    def get_bulk_from_data_save(self, dic):
        data = copy.deepcopy(dic)
        Z = torch.zeros((2,data['X'].shape[1],data['X'].shape[2])) # Kmax x N x dim
        Ndrug = data['X'].shape[1]
        weights = data['proportions'].T[:,:Ndrug] # Kmax x N
        weights = weights / (torch.sum(weights, dim=0)[None,:])
        Z[0,:,:] = torch.sum(torch.nan_to_num(data['X']) * weights[:,:,None], dim=0)
        Z[1,:,:] = copy.deepcopy(Z[0,:,:])
        data['X'] = copy.deepcopy(Z)
        props = torch.zeros((data['n_c'].shape[1],2))
        props[:,0] = torch.mean(data['n0_c']/data['n_c'], dim=0)
        props[:,1] = 1-props[:,0]
        data['ini_proportions'] = copy.deepcopy(props)
        data['Kmax'] = 2
        props = torch.zeros((2,data['n_c'].shape[1]))
        props[0,:] = data['proportions'][:,0]
        props[1,:] = 1-props[0,:]
        data['proportions'] = copy.deepcopy(props.T)

        data['n_rna'] = None
        data['masks']['RNA'] = torch.tensor([0])

        self.cluster2clonelabel = ['healthy','tumor']
        self.clonelabel2cat = {'healthy':'healthy', 'tumor':'tumor'}
        self.init_cat_clonelabel()
        return data
    
    def get_bulk_from_data(self, dic):
        data = copy.deepcopy(dic)
        self.cluster2clonelabel = ['healthy','tumor']
        self.clonelabel2cat = {'healthy':'healthy', 'tumor':'tumor'}
        self.init_cat_clonelabel()
        Z = torch.zeros((2,data['X'].shape[1],data['X'].shape[2])) # Kmax x N x dim
        Ndrug = data['X'].shape[1]
        weights = data['proportions'].T[:,:Ndrug] # Kmax x N
        weights = weights / (torch.sum(weights, dim=0)[None,:])
        Z[0,:,:] = torch.sum(torch.nan_to_num(data['X']) * weights[:,:,None], dim=0)
        Z[1,:,:] = copy.deepcopy(Z[0,:,:])
        data['X'] = copy.deepcopy(Z)
        props = torch.zeros((data['n_c'].shape[1],2))
        props[:,0] = torch.mean( (data['n0_c']/data['n_c'])[self.cat2clusters['healthy'],:], dim=0)
        props[:,1] = 1-props[:,0]
        data['ini_proportions'] = copy.deepcopy(props)
        data['Kmax'] = 2
        props = torch.zeros((2,data['n_c'].shape[1]))
        props[0,:] = torch.sum(data['proportions'][:,self.cat2clusters['healthy']], dim=1)
        props[1,:] = 1-props[0,:]
        data['proportions'] = copy.deepcopy(props.T)

        data['n_rna'] = None
        data['masks']['RNA'] = torch.tensor([0])

        
        return data

    
    def get_bimodal_from_data(self, dic):
        data = copy.deepcopy(dic)
        self.cluster2clonelabel = ['healthy','tumor']
        self.clonelabel2cat = {'healthy':'healthy', 'tumor':'tumor'}
        self.init_cat_clonelabel()
        Z = torch.zeros((2,data['X'].shape[1],data['X'].shape[2])) # Kmax x N x dim
        Ndrug = data['X'].shape[1]
        weights = data['proportions'].T[self.cat2clusters['healthy'],:Ndrug] # Kmax x N
        weights = weights / (torch.sum(weights, dim=0)[None,:])
        Z[0,:,:] = torch.sum( torch.nan_to_num(data['X'][self.cat2clusters['healthy'],:,:]) * weights[:,:,None], dim=0)
        weights = data['proportions'].T[self.cat2clusters['tumor'],:Ndrug] # Kmax x N
        weights = weights / (torch.sum(weights, dim=0)[None,:])
        Z[1,:,:] = torch.sum( torch.nan_to_num(data['X'][self.cat2clusters['tumor'],:,:]) * weights[:,:,None], dim=0)
        data['X'] = copy.deepcopy(Z)
        props = torch.zeros((data['n_rna'].shape[1],2))  
        RNA = torch.nan_to_num(data['n_rna'])
        props[:,0] = torch.sum(RNA[self.cat2clusters['healthy'],:], dim=0) / torch.sum(RNA, dim=0)
        props[:,1] = 1-props[:,0]
        data['ini_proportions'] = copy.deepcopy(props)

        data['masks']['RNA'] = torch.full((2,data['N']), True)
        rna = torch.zeros((2,data['n_rna'].shape[1]))  
        rna[0,:] = torch.sum(data['n_rna'][self.cat2clusters['healthy'],:], dim=0)
        rna[1,:] = torch.sum(data['n_rna'][self.cat2clusters['tumor'],:], dim=0)
        data['n_rna'] = copy.deepcopy(rna)
        data['Kmax'] = 2
        props = torch.zeros((2,data['n_c'].shape[1]))
        props[0,:] = torch.sum(data['proportions'][:,self.cat2clusters['healthy']], dim=1)
        props[1,:] = 1-props[0,:]
        data['proportions'] = copy.deepcopy(props.T)
        
        return data
    
    
    def get_bimodal_from_data_save(self, dic):
        data = copy.deepcopy(dic)
        Z = torch.zeros((2,data['X'].shape[1],data['X'].shape[2])) # Kmax x N x dim
        Z[0,:,:] = data['X'][0,:,:]
        Ndrug = data['X'].shape[1]
        weights = data['proportions'].T[1:,:Ndrug] # Kmax x N
        weights = weights / (torch.sum(weights, dim=0)[None,:])
        Z[1,:,:] = torch.sum(data['X'][1:,:,:] * weights[:,:,None], dim=0)
        data['X'] = copy.deepcopy(Z)
        props = torch.zeros((data['n_rna'].shape[1],2))  
        props[:,0] = data['n_rna'][0,:] / torch.sum(data['n_rna'], dim=0)
        props[:,1] = 1-props[:,0]
        data['ini_proportions'] = copy.deepcopy(props)

        data['masks']['RNA'] = torch.full((2,data['N']), True)
        rna = torch.zeros((2,data['n_rna'].shape[1]))  
        rna[0,:] = data['n_rna'][0,:]
        rna[1,:] = torch.sum(data['n_rna'][1:,:], dim=0)
        data['n_rna'] = copy.deepcopy(rna)
        data['Kmax'] = 2
        props = torch.zeros((2,data['n_c'].shape[1]))
        props[0,:] = data['proportions'][:,0]
        props[1,:] = 1-props[0,:]
        data['proportions'] = copy.deepcopy(props.T)
        self.cluster2clonelabel = ['healthy','tumor']
        self.clonelabel2cat = {'healthy':'healthy', 'tumor':'tumor'}
        self.init_cat_clonelabel()
        return data
    
    def over_sample(self, dic, multi_C = 4, multi_R = 4):
        DIC = copy.deepcopy(dic)
        DIC['R'] = multi_R * dic['R']
        DIC['masks']['R'] =  torch.repeat_interleave(dic['masks']['R'], multi_R, axis=0)
        DIC['n_r'] = torch.repeat_interleave(dic['n_r'], multi_R, axis=0)
        DIC['X_nu_drug'] = torch.repeat_interleave(dic['X_nu_drug'], multi_R, axis=0)
        DIC['C'] = multi_C * dic['C']
        DIC['masks']['C'] =  torch.repeat_interleave(dic['masks']['C'], multi_C, axis=0)
        DIC['n_c'] = torch.repeat_interleave(dic['n_c'], multi_C, axis=0)
        DIC['X_nu_control'] = torch.repeat_interleave(dic['X_nu_control'], multi_C, axis=0)
        DIC_sample = self.sampling(DIC)
        DIC = load_from_sampling(DIC, DIC_sample)
        return DIC, DIC_sample