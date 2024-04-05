import torch
import numpy as np
import copy
from server.avgserver import Avg_Server

class Fedprophet_Avg_Server(Avg_Server):
    def __init__(self, global_model, clients, selector, scheduler, 
                 stat_monitor, sys_monitor, frac=None, weights=None, 
                 test_dataset=None, device=torch.device('cpu'), test_every=1, **kwargs):
        
        super().__init__(global_model, clients, selector, scheduler, 
                        stat_monitor, sys_monitor, frac, weights, 
                        test_dataset, device, test_every)
        

        assert hasattr(scheduler,"auxiliary_model_dict"), "The scheduler must have a dict of auxiliary model!"
        for m in scheduler.auxiliary_model_dict:
            if scheduler.auxiliary_model_dict[m] is not None:
                scheduler.auxiliary_model_dict[m].to(device)
        self.global_model["aux_models"] = scheduler.auxiliary_model_dict


    def aggregate(self,local_models,model,aux_models,training_hyperparameters,**kwargs):
        # aggregate backbone model
        masked_weights = [None]*self.num_users
        for n,lm in enumerate(local_models):
            if lm is not None:
                local_model = lm["model"]
                local_model.to(self.device)
                lw = local_model.state_dict()
                masked_weights[n] = copy.deepcopy(lw)
                module_list = []
                trained_module_list = training_hyperparameters[n]["stage_module_list"] \
                            + training_hyperparameters[n]["prophet_module_list"]
                for m in trained_module_list:
                    module_list += m.split("+")
                for p in lw: # mask out the parameters that have not been trained
                    mask_out = True
                    for m in module_list:
                        if m+"." in p:
                            mask_out = False
                            break

                    if mask_out: # drop out the layers that is not trained
                        masked_weights[n].pop(p)
                
        
        model = copy.deepcopy(model)
        w0 = model.state_dict()
        for p in w0.keys():
            weights_sum = 0.0
            w = 0
            for n,lw in enumerate(masked_weights):
                if lw is not None and p in lw:
                    weights_sum += self.weights[n]
                    w += self.weights[n]*lw[p]
            if weights_sum > 0:
                w0[p] = w/weights_sum
        model.load_state_dict(w0)

        # aggregate auxiliary model
        aux_models = copy.deepcopy(aux_models)
        # aggregate stage auxiliary model
        local_aux_models = {m: np.array([None for _ in range(self.num_users)]) \
                             for m in aux_models.keys()}
        for n,lm in enumerate(local_models):
            if lm is not None:
                stage_aux_model = lm["stage_aux_model"]
                prophet_aux_model = lm["prophet_aux_model"]
                if stage_aux_model is not None:
                    stage_aux_model.to(self.device)
                    current_stage = training_hyperparameters[n]["stage_aux_model_name"]
                    local_aux_models[current_stage][n] = stage_aux_model
                if prophet_aux_model is not None:
                    prophet_aux_model.to(self.device)
                    prophet_stage = training_hyperparameters[n]["prophet_aux_model_name"]
                    local_aux_models[prophet_stage][n] = prophet_aux_model
                    
        for m in aux_models:
            if aux_models[m] is not None:
                aux_models[m] = super().aggregate(local_aux_models[m],aux_models[m])["model"]

        return {"model":model,"aux_models":aux_models}
