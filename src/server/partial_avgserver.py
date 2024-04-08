import torch
import copy
import numpy as np
from server.avgserver import Avg_Server

class Partial_Avg_Server(Avg_Server):
    def __init__(self, global_model, clients, selector, scheduler, 
                 stat_monitor, sys_monitor, frac=None, weights=None, 
                 test_dataset=None, device=torch.device('cpu'), 
                 test_every=1, **kwargs):
        super().__init__(global_model, clients, selector, scheduler,
                          stat_monitor, sys_monitor, frac, weights, 
                          test_dataset, device, test_every)
        
    def aggregate(self, local_models, model, training_hyperparameters=None,
                   **kwargs):
        """
        The key idea is that the parts without update will not be counted 
        into the weights of aggregation.
        """
        model = copy.deepcopy(model)
        w0 = model.state_dict()
        for p in w0.keys():
            weights_sum = torch.zeros_like(w0[p])
            w = 0
            for lm in local_models:
                if lm is not None:
                    local_model = lm["model"]
                    local_model.to(self.device)
                    lw = local_model.state_dict()
                    local_update_idxs = lm["updated_partial_idxs"]
                    if p in lw and p in local_update_idxs:
                        update = lw[p]-w0[p]
                        updated_idx = local_update_idxs[p].to(self.device)
                        weights_sum += updated_idx
                        w += update*updated_idx
            if torch.sum(weights_sum) > 0:
                w0[p] += w/(weights_sum+1e-5)
        model.load_state_dict(w0)
        return {"model":model}
                