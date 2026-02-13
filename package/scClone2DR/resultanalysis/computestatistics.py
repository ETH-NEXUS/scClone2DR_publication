import torch
import copy
import numpy as np
from ..utils import *


def KL(a,b, mode='survival'):
    res = 0
    assert (torch.sum(a<0)==0)
    assert (torch.sum(b<0)==0)
    assert (torch.sum(a>1)==0)
    assert (torch.sum(b>1)==0)
    if mode=='survival':
        res = a*np.log(a/b) + (1-a)*np.log((1-a)/(1-b))
        
    else:
        res = a*np.log(a/b)
    res = torch.nan_to_num(res)
    return torch.sum(res)


class ComputeStatistics():
    def __init__(self):
        self.results = {}

    def compute_all_stats(self, true_params, data, params):
        params_tensor = self.convert_to_tensor(params)
        self.compute_KL_proportions(true_params, params_tensor)
        self.compute_KL_survival_proba(true_params, params_tensor)
        self.compute_spearman_drug(true_params, data, params=params_tensor)
        self.compute_spearman_subclone(true_params, data, params=params_tensor)
        self.compute_beta_L2_error(true_params, params_tensor)
        self.compute_drug_effects(params_tensor, true_params=true_params)
        self.compute_error_overall_survival(true_params, params_tensor)
        self.get_fold_change(data, params, true_params=true_params)
        
    def compute_all_stats4bulk_or_bimodal(self, model, data_ref, databulk, params_svi_bulk):
        data_ref['pi'] = model.compute_survival_probas_subclone_features(data_ref, data_ref)
        params_svi_bulk_tensor = self.convert_to_tensor(params_svi_bulk)
        pi_bulk_2 = self.compute_survival_probas_subclone_features(databulk, params_svi_bulk_tensor)
        D, _, N = pi_bulk_2.shape
        pi_bulk = torch.zeros((D, data_ref['Kmax'], N))
        pi_bulk[:,0,:] = pi_bulk_2[:,0,:]
        pi_bulk[:,1:,:] = pi_bulk_2[:,1,:].unsqueeze(1)
        
        props_bulk = torch.cat((params_svi_bulk_tensor['proportions'], torch.zeros((data_ref['N'], data_ref['Kmax']-2))), dim=1)
        
        self.compute_KL_survival_proba(data_ref, {'pi':pi_bulk})
        self.compute_error_overall_survival(data_ref, {'pi':pi_bulk, 'proportions':props_bulk})
        true_scores, hat_scores, spearman_drugs_avg = model.compute_spearman_drug(data_ref, data_ref, params={'pi':pi_bulk}, output_results=True)
        self.results['drug_scores'] = hat_scores
        self.results['true_drug_scores'] = true_scores 
        self.results['spearman_drugs_avg'] = spearman_drugs_avg 
        scores, true_scores, L1err_scores = model.compute_drug_effects({'pi':pi_bulk, 'proportions':props_bulk}, true_params=data_ref, output_results=True)
        self.results['drug_effects'] = scores
        self.results['true_drug_effects'] = true_scores
        self.results['L1err_drug_effects'] = L1err_scores

        all_fold_changes, _, _, colors = model.get_fold_change(data_ref, {'pi':pi_bulk, 'proportions':props_bulk}, true_params=data_ref, output_results=True)
        self.results['fold_change_true'] = all_fold_changes['true']
        self.results['fold_change_pred'] = all_fold_changes['pred']
        self.results['fold_change_data'] = all_fold_changes['not pred']

    def merge_data_param(self, dic, params):
        data = {}
        for key, val in dic.items():
            if torch.is_tensor(val):
                try:
                    data[key] = val.clone().detach()
                except:
                    data[key] =  copy.deepcopy(val)
            else:
                 data[key] =  copy.deepcopy(val)
        if not(params is None): 
            for key,val in params.items():
                if isinstance(val, np.ndarray):  # Check if the value is a numpy array
                    try:
                        data[key] = torch.tensor(val)
                    except:
                        data[key] =  copy.deepcopy(val)
                else:
                    data[key] = val
        return data
        
    def compute_KL_proportions(self, true_params, params):
        """
        Compute the Kullback-Leibler divergence beteween the true proportions of clones and the estimated ones.
        """
        self.results['KL_props'] = KL(params['proportions'], true_params['proportions'], mode='proportions') / true_params['proportions'].shape[0]
        self.results['proportions'] = params['proportions']
        self.results['true_proportions'] = true_params['proportions']
    
    def compute_KL_survival_proba(self, true_params, params):
        """
        Compute the Kullback-Leibler divergence beteween the true survival probabilities of clones and the estimated ones.
        """
        D,Kmax,N = params['pi'].shape
        pi = copy.deepcopy(params['pi'])
        for d in range(D):
            for i in range(N):
                coeff = torch.mean(true_params['pi'][d,:,i]/params['pi'][d,:,i])
                pi[d,:,i] *= min([coeff,1/torch.max(params['pi'][d,:,i])])
        self.results['KL_survival_probas'] = KL(pi, true_params['pi'], mode='survival') / (D*N)
        self.results['survival_probas'] = pi
        self.results['true_survival_probas'] = true_params['pi']

    def compute_spearman_drug(self, true_params, dic, params=None, output_results=False):
        data = self.merge_data_param(dic, params)
        from scipy.stats import spearmanr
        spearman_drugs_avg = 0
        hat_scores = []
        true_scores = []
        for i in range(data['N']):
            hat_score = torch.min(data['pi'][:,self.cat2clusters['healthy'],i], dim=1)[0] / torch.max(data['pi'][:,self.cat2clusters['tumor'],i], dim=1)[0]
            true_score = torch.min(true_params['pi'][:,self.cat2clusters['healthy'],i], dim=1)[0] / torch.max(true_params['pi'][:,self.cat2clusters['tumor'],i], dim=1)[0]
            spearman_drugs_avg += spearmanr(hat_score,true_score)[0] / (data['N'])  
            hat_scores += list(hat_score)
            true_scores += list(true_score)
        if output_results:
            return true_scores, hat_scores, spearman_drugs_avg
        else:
            self.results['drug_scores'] = hat_scores
            self.results['true_drug_scores'] = true_scores 
            self.results['spearman_drugs_avg'] = spearman_drugs_avg

    def compute_error_overall_survival(self, true_params, params):
        D,Kmax,N = params['pi'].shape
        pi = copy.deepcopy(params['pi'])
        for d in range(D):
            for i in range(N):
                coeff = torch.mean(true_params['pi'][d,:,i]/params['pi'][d,:,i])
                pi[d,:,i] *= min([coeff,1/torch.max(params['pi'][d,:,i])])
        proba_tot = np.zeros((D,N))
        true_proba_tot = np.zeros((D,N))
        for d in range(D):
            for i in range(N):
                proba_tot[d,i] = torch.sum(pi[d,:,i]*params['proportions'])
                true_proba_tot[d,i] = torch.sum(true_params['pi'][d,:,i]*true_params['proportions'])
        self.results['L1err_overall_survival'] = np.mean(np.abs(proba_tot-true_proba_tot))
        
    def compute_spearman_subclone(self, true_params, dic, params=None):
        data = self.merge_data_param(dic, params)
        from scipy.stats import spearmanr
        self.results['spearman_subclones_avg'] = 0
        for i in range(data['N']):
            for d in range(data['D']):
                hat_score = torch.mean(data['pi'][d,self.cat2clusters['healthy'],i]) / data['pi'][d,self.cat2clusters['tumor'],i]
                true_score = torch.mean(true_params['pi'][d,self.cat2clusters['healthy'],i]) / true_params['pi'][d,self.cat2clusters['tumor'],i]
                self.results['spearman_subclones_avg'] += spearmanr(hat_score,true_score)[0] /  (data['N']*data['D'])

    def compute_beta_L2_error(self, true_params, params):
        """
        Compute the L2 error beteween the true beta matrix and the estimated one.
        """
        self.results['MSE_beta'] = torch.norm(params['beta']-true_params['beta'])**2 / np.prod(true_params['beta'].shape)

    def compute_drug_effects(self, params, true_params=None, output_results=False):
        L1err_scores = None
        true_scores = None
        scores = torch.sum(params['pi'][:,self.cat2clusters['tumor'],:] * (params['proportions'].T)[None,self.cat2clusters['tumor'],:], dim=1)
        scores /= torch.sum(params['pi'] * (params['proportions'].T)[None,:,:], dim=1)
        scores /= (1-torch.mean((params['proportions'].T)[None,self.cat2clusters['healthy'],:], dim=1))
        scores = scores.reshape(-1)
        if not(true_params is None):
            true_scores = torch.sum(true_params['pi'][:,self.cat2clusters['tumor'],:] * (true_params['proportions'].T)[None,self.cat2clusters['tumor'],:], dim=1)
            true_scores /= torch.sum(true_params['pi'] * (true_params['proportions'].T)[None,:,:], dim=1)
            true_scores /= (1-torch.mean((true_params['proportions'].T)[None,self.cat2clusters['healthy'],:], dim=1))
            true_scores = true_scores.reshape(-1)
            L1err_scores = torch.mean(torch.abs(true_scores-scores))
        if output_results:
            return scores, true_scores, L1err_scores
        else:
            self.results['drug_effects'] = scores
            self.results['true_drug_effects'] = true_scores
            self.results['L1err_drug_effects'] = L1err_scores

    def get_fold_change(self, data, params_svi, true_params=None, DIC_sample=None, idxs_filtering_samples=None, output_results=False, drug_filter=None, color_mode=None):        
        
        data_esti = self.merge_data_param(data, params_svi)

        if (true_params is None) and ('pi' in params_svi.keys()):
            DIC_sample = params_svi
        elif DIC_sample is None:
            _, DIC_sample = self.over_sample(data_esti, multi_C = 1, multi_R = 1)
            DIC_sample = load_from_sampling(DIC_sample, data)
        DIC = data_esti

        preds_c = self.get_mean_fracMEL_control(DIC['proportions'].T, DIC['C'], DIC_sample['nu_healthy_control'])
        preds_c = (1-preds_c)

        PI = DIC_sample['pi']

        preds_drugs = self.get_mean_fracMEL_treated(DIC['proportions'].T, DIC['D'], PI, DIC_sample['nu_healthy_drug'])
        preds_drugs = (1-preds_drugs)
        onesPI = np.ones(DIC_sample['pi'].shape)
        preds_c4drugs = self.get_mean_fracMEL_treated(DIC['proportions'].T, DIC['D'], onesPI, DIC_sample['nu_healthy_drug'])
        preds_c4drugs = (1-preds_c4drugs)

        if not(true_params is None):
            data_true = data.copy()
            for key, val in true_params.items():
                data_true[key] = val

            DIC_true, DIC_true_sample = self.over_sample(data_true, multi_C = 1, multi_R = 1)
            DIC_true_sample = load_from_sampling(DIC_true_sample, data)

            truePI = DIC_true_sample['pi']
            preds_drugs_true = self.get_mean_fracMEL_treated(DIC_true['proportions'].T, DIC_true['D'], truePI, DIC_true_sample['nu_healthy_drug'])
            preds_drugs_true = (1-preds_drugs_true)
            onesPI = np.ones(DIC_true_sample['pi'].shape)
            preds_c4drugs_true = self.get_mean_fracMEL_treated(DIC_true['proportions'].T, DIC_true['D'], onesPI, DIC_true_sample['nu_healthy_drug'])
            preds_c4drugs_true = (1-preds_c4drugs_true)


            preds_c_true = self.get_mean_fracMEL_control(DIC_true['proportions'].T, DIC_true['C'], DIC_true_sample['nu_healthy_control'])
            preds_c_true = (1-preds_c_true)

            ls_modes = ['not pred', 'pred', 'true']
        else:
            ls_modes = ['not pred', 'pred']
            truePI = None

            
        if drug_filter is None:
            drug_filter = np.array([i for i in range(DIC['D'])])


        def get_robust(ls):
            ls = np.sort(ls)
            n = len(ls)
            return ls[:n]

        from scipy import stats as stats

        USE_LEARNT_PARAMS = True

        if idxs_filtering_samples is None:
            idxs_filtering_samples = [i for i in range(DIC['N'])]

        all_color_modes = ['', 'drug', 'patient', 'train_test']
        if color_mode is None:
            color_mode = all_color_modes[1]
        hsv = plt.get_cmap('jet')
        if color_mode=='drug':
            ref_colors = hsv(np.linspace(0, 1.0, len(drug_filter)))
        elif color_mode=='sample':
            ref_colors = hsv(np.linspace(0, 1.0, len(idxs_filtering_samples)))
        elif 'patient_id':
            all_patient_ids = self.dfinfo['patient_id'].values
            unique_labels = np.unique(self.dfinfo['patient_id'].values)
            ref_colors = hsv(np.linspace(0, 1.0, len(unique_labels)))
        else:
            colors = np.array([['blue' for d in range(len(drug_filter))] for i in range(len(idxs_filtering_samples))]).reshape(len(drug_filter)*len(idxs_filtering_samples))

        ls_stat_tests = ['mannwhitneyu', 't_test', 'Welch']
        stat_test = ls_stat_tests[1]

        all_fold_changes = {}

        for mode in ls_modes:
            count = 0
            colors = []
            patient_pvals = []
            patient_fold_changes = []
            for patient_id in idxs_filtering_samples:
                for drug_id in drug_filter:
                    #### P VALUE
                    nb_replis = torch.sum(DIC['masks']['R'][:,drug_id, patient_id])
                    nb_replis_c = torch.sum(DIC['masks']['C'][:, patient_id])
                    patient_control_data = (DIC['n0_c']/DIC['n_c'])[:nb_replis_c, patient_id]
                    if mode=='not pred':
                        patient_drug_data = (DIC['n0_r']/DIC['n_r'])[:nb_replis,drug_id, patient_id]
                        patient_control_data = get_robust(patient_control_data)
                    elif mode=='pred':
                        if USE_LEARNT_PARAMS:
                            patient_control_data = preds_c4drugs[:nb_replis_c, drug_id, patient_id]
                        patient_drug_data = preds_drugs[:nb_replis,drug_id, patient_id]
                    else:
                        patient_control_data = preds_c4drugs_true[:nb_replis_c, drug_id, patient_id]
                        patient_drug_data = preds_drugs_true[:nb_replis,drug_id, patient_id]

                    if stat_test=="mannwhitneyu":
                        t_statistic, pval = stats.mannwhitneyu(np.log(patient_control_data), np.log(patient_drug_data))
                    elif stat_test=='t_test':
                        t_statistic, pval = stats.ttest_ind(np.log(patient_control_data), np.log(patient_drug_data))
                    elif stat_test=='Welch':
                        t_statistic, pval = stats.ttest_ind(np.log(patient_control_data), np.log(patient_drug_data), equal_var=False)
                    patient_pvals.append(pval)

                    #### FOLD CHANGE
                    if USE_LEARNT_PARAMS and mode=='pred':
                        patient_control_data = torch.log(preds_c[:nb_replis_c, patient_id])
                        patient_drug_data = torch.log(preds_drugs[:nb_replis,drug_id, patient_id])
                        fold_change = (get_robust(patient_control_data)).mean() - patient_drug_data.mean()
                    elif mode=='not pred':
                        patient_drug_data = torch.log(DIC['n0_r']/DIC['n_r'])[:nb_replis,drug_id, patient_id]
                        patient_control_data = torch.log(DIC['n0_c']/DIC['n_c'])[:nb_replis_c, patient_id]
                        fold_change = (get_robust(patient_control_data)).mean() - patient_drug_data.mean()
                    else:
                        patient_control_data = torch.log(preds_c_true[:nb_replis_c, patient_id])
                        patient_drug_data = torch.log(preds_drugs_true[:nb_replis,drug_id, patient_id])
                        fold_change = (get_robust(patient_control_data)).mean() - patient_drug_data.mean()

                    patient_fold_changes.append(fold_change)

                    if color_mode=="sample":
                        colors.append(ref_colors[patient_id])
                    elif color_mode=="patient_id":
                        colors.append(ref_colors[all_patient_ids[patient_id]])
                    elif color_mode=="drug":
                        colors.append(ref_colors[drug_id])
                    elif color_mode=='train_test':
                        if patient_id<DIC['n_r'].shape[2]:
                            colors.append('green')
                        else:
                            colors.append('red')

                    count += 1

            all_fold_changes[mode] = patient_fold_changes

        if output_results:
            if not(true_params is None):
                return all_fold_changes, PI, truePI, colors
            else:
                return all_fold_changes, PI, colors
        else:
            if not(true_params is None):
                self.results['fold_change_true'] = all_fold_changes['true']
            self.results['fold_change_pred'] = all_fold_changes['pred']
            self.results['fold_change_data'] = all_fold_changes['not pred']
