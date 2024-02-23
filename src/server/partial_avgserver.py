import torch
from server.avgserver import Avg_Server

class Partial_Avg_Server(Avg_Server):
    def __init__(self, global_model, clients, selector, scheduler, 
                 stat_monitor, sys_monitor, frac=None, weights=None, 
                 test_dataset=None, device=torch.device('cpu'), 
                 test_every=1, **kwargs):
        super().__init__(global_model, clients, selector, scheduler,
                          stat_monitor, sys_monitor, frac, weights, 
                          test_dataset, device, test_every, **kwargs)
        