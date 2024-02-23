import torch
import numpy as np
import copy
from server.avgserver import Avg_Server

class Fedprophet_Avg_Server(Avg_Server):
    def __init__(self, global_model, clients, selector, scheduler, 
                 stat_monitor, sys_monitor, frac=None, weights=None, 
                 test_dataset=None, device=torch.device('cpu'), 
                 test_every=1, **kwargs):
        
        
        self.device = device
        global_model.to(device)
        self.global_model = {"model":global_model}

        assert hasattr(scheduler,"auxiliary_model_dict"), "The scheduler must have a dict of auxiliary model!"
        for m in scheduler.auxiliary_model_dict:
            if scheduler.auxiliary_model_dict[m] is not None:
                scheduler.auxiliary_model_dict[m].to(device)
        self.global_model["aux_models"] = scheduler.auxiliary_model_dict

        self.clients = clients
        self.num_users = len(self.clients)
        self.train_frac = frac
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones(self.num_users)/self.num_users
        self.test_dataset = test_dataset

        self.selector = selector
        self.scheduler = scheduler
        self.stat_monitor = stat_monitor
        self.sys_monitor = sys_monitor

        
        self.round = 0
        self.idxs_users = None
        self.test_every = test_every

        # collect the init loss and training latency
        
        self.stat_info = self.val(self.global_model)
        self.sys_info = self.sys_monitor.collect()
        
        

    def train_idx(self,idxs_users):
        local_weights = np.array([None for _ in range(self.num_users)])
        module_lists = [[] for _ in range(self.num_users)]
        auxiliary_weights = {m: np.array([None for _ in range(self.num_users)]) \
                             for m in self.global_model["aux_models"].keys()}
        for idx in idxs_users:
            training_hyperparameters = self.scheduler.training_params(idx=idx,chosen_idxs=idxs_users)
            models = self.clients[idx].train(**self.global_model,**training_hyperparameters)
            assert len(models)>=3, \
            "The clients must return backbone model, stage auxiliary model and prophet auxiliary model at the same time!"
            local_model,stage_aux_model,prophet_aux_model,*_ = models
            # return the local model to the server's device
            local_model.to(self.device) 
            local_weights[idx] = copy.deepcopy(local_model.state_dict())
            if stage_aux_model is not None:
                stage_aux_model.to(self.device)
                current_stage = training_hyperparameters["stage_aux_model_name"]
                auxiliary_weights[current_stage][idx] = copy.deepcopy(stage_aux_model.state_dict())
            if prophet_aux_model is not None:
                prophet_aux_model.to(self.device)
                prophet_stage = training_hyperparameters["prophet_aux_model_name"]
                auxiliary_weights[prophet_stage][idx] = copy.deepcopy(prophet_aux_model.state_dict())
            
            trained_module_list = training_hyperparameters["stage_module_list"] \
                            + training_hyperparameters["prophet_module_list"]
            for m in trained_module_list:
                module_lists[idx] += m.split("+")
            

                
            
        global_model,aux_models = self.aggregate(weights = local_weights,
                                                 init_model = self.global_model["model"],
                                                 aux_weights = auxiliary_weights,
                                                 init_aux_models = self.global_model["aux_models"],
                                                 module_lists = module_lists)

        return {"model":global_model,"aux_models":aux_models}

    def aggregate(self,weights,init_model,aux_weights,init_aux_models,module_lists,**kwargs):
        # aggregate backbone model
        masked_weights = copy.deepcopy(weights)
        for n,lw in enumerate(weights):
            if lw is not None:
                for p in lw:
                    mask_out = True
                    for m in module_lists[n]:
                        if m+"." in p:
                            mask_out = False
                            break

                    if mask_out: # drop out the layers that is not trained
                        masked_weights[n].pop(p)

        model = super().aggregate(masked_weights,init_model)

        # aggregate auxiliary model
        aux_models = copy.deepcopy(init_aux_models)
        for m in aux_weights:
            if aux_models[m] is not None:
                aux_models[m] = super().aggregate(aux_weights[m],init_aux_models[m])

        return model,aux_models
