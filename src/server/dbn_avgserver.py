import torch
import torch.nn.functional as F
import copy
import numpy as np
from server.avgserver import Avg_Server
from types import MethodType

class DBN_Avg_Server(Avg_Server):
    def __init__(self, global_model, clients, selector, scheduler, 
                 stat_monitor, sys_monitor, frac=None, weights=None, 
                 test_dataset=None, device=torch.device('cpu'), 
                 test_every=1, **kwargs):
        super().__init__(global_model, clients, selector, scheduler,
                          stat_monitor, sys_monitor, frac, weights, 
                          test_dataset, device, test_every)
        
    def train_idx(self,idxs_users):
        local_models = np.array([None for _ in range(self.num_users)])
        local_training_params = np.array([None for _ in range(self.num_users)])
        local_states = np.array([None for _ in range(self.num_users)])
        ST_idx = []
        AT_idx = []
        for idx in idxs_users:
            training_hyperparameters = self.scheduler.training_params(idx=idx,chosen_idxs=idxs_users)
            local_model = self.clients[idx].train(**self.global_model,**training_hyperparameters)
            local_models[idx] = local_model
            local_training_params[idx] = training_hyperparameters
            local_states[idx] = self.clients[idx].local_states
            if training_hyperparameters['adv_train']:
                AT_idx.append(idx)
            else:
                ST_idx.append(idx)

        global_model = self.aggregate(local_models = local_models,
                                      training_hyperparameters = local_training_params,
                                      **self.global_model)
        
        local_states = self.robustness_propagation(local_states,ST_idx,AT_idx)
        for idx in idxs_users:
            self.clients[idx].local_states = local_states[idx]
        return global_model
    
    def robustness_propagation(self,local_states,STidx,ATidx,T=0.01):
        if not ATidx:
            return local_states
        for t in STidx:
            mu_r = [] # shape as S x L, where S is the number of AT clients, and L is the number of layers
            sigmasquare_r = [] # same as above
            Similarity = [] # shape as S
            
            for s in ATidx:
                L = 0
                mean_stat_name = []
                var_stat_name = []

                Ds = 0 # q_j in the paper
                mu_rs = [] # mu_s in each layer
                sigmasquare_rs = [] # sigmasquare_s in each layer
                for state in local_states[s].keys():
                    if 'mean' in state and 'adv' not in state:
                        L+=1
                        var_state = state.replace('mean','var')
                        adv_mean_state = state.replace('running','adv_running')
                        adv_var_state = var_state.replace('running','adv_running')
                        mean_stat_name.append(adv_mean_state)
                        var_stat_name.append(adv_var_state)

                        
                        Ds += 0.5*(F.cosine_similarity(local_states[t][state],local_states[s][state],dim=0)+\
                            F.cosine_similarity(local_states[t][var_state],local_states[s][var_state],dim=0))

                        mu_rs.append(local_states[s][adv_mean_state])
                        sigmasquare_rs.append(local_states[s][adv_var_state])
                mu_r.append(mu_rs)
                sigmasquare_r.append(sigmasquare_rs)
                Ds /= (L*T) 
                Similarity.append(Ds)
                
            

            Similarity = torch.tensor(Similarity)
            alphas = torch.softmax(Similarity,dim=0)
            
            for l in range(L):
                mu = 0.0
                var = 0.0
                for n,alpha in enumerate(alphas): # average mu_r from each source AT client with their alpha weighted
                    mu += alpha*mu_r[n][l]
                    var += alpha*sigmasquare_r[n][l]
                local_states[t][mean_stat_name[l]] = mu
                local_states[t][var_stat_name[l]] = var
                
        
        
        return local_states
                
    def load(self,save_file):
        # make the class consistent with the original saved model
        def clean(self):
            return
        def adv(self):
            return
        type(self.global_model["model"]).clean = MethodType(clean, type(self.global_model["model"]))
        type(self.global_model["model"]).adv = MethodType(adv, type(self.global_model["model"]))
        super().load(save_file)