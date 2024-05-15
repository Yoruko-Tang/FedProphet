import os.path as osp
import time
import datetime
import json
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from copy import deepcopy
from types import MethodType

from client import *
from server import *
from selection import *
from monitor import *
from scheduler import *


from utils.options import args_parser
from models.model_utils import get_net
from utils.utils import setup_seed, get_log_path, exp_details

from datasets.dataset_utils import get_dataset, get_data_matrix
from datasets import dataset_to_datafamily

from hardware.sys_utils import get_devices, model_summary






if __name__ == '__main__':

    start_time = time.time()
    # path_project = os.path.abspath('..')
    args = args_parser()
    params = vars(args)
    params["date"]=str(datetime.datetime.now())
    exp_details(args)
    base_file = get_log_path(args)
    
    device = torch.device('cuda:'+args.gpu if args.gpu else 'cpu')
    
    if args.gpu:
        torch.cuda.set_device(device)

    for seed in args.seed: ## fix seed
        file_name = osp.join(base_file,'seed'+str(seed))

        print("Start with Random Seed: {}".format(seed))
        
        ## ==================================Load Dataset==================================
        train_dataset, test_dataset, user_groups, weights = get_dataset(args,seed)
        data_matrix = get_data_matrix(train_dataset, user_groups, args.num_classes)
        user_devices = get_devices(args,seed)
        if seed is not None:
            setup_seed(seed)


        ## ==================================Build Model==================================
        # try to get the model from torchvision.models with pretraining
        global_model = get_net(modelname = args.model_arch,
                               modeltype = dataset_to_datafamily[args.dataset],
                               num_classes = args.num_classes,
                               pretrained = False,
                               norm_type = args.norm,
                               adv_norm = (args.adv_train or args.adv_test),
                               modularization = (args.flalg == "FedProphet"),
                               partialization = (args.flalg in ["HeteroFL",'FedDrop',"FedRolex"]))
        
        # make the class consistent with the original saved model
        if args.flalg == "FedProphet": 
            def module_forward(self, x, module_list):
                return None
            type(global_model).module_forward = MethodType(module_forward, type(global_model))
        elif args.flalg in ["HeteroFL",'FedDrop',"FedRolex"]:
            def partial_forward(self,x, neuron_dict):
                return None
            type(global_model).partial_forward = MethodType(partial_forward, type(global_model))

        model_profile = model_summary(model = deepcopy(global_model),
                                      inputsize=[args.local_bs]+list(train_dataset[0][0].shape),
                                      default_local_eps=args.local_ep,
                                      **vars(args))
        
        

        ## ==================================Build Clients==================================
        
        clients = [AT_Client(train_dataset,user_groups[i],
                            sys_info=user_devices[i],
                            model_profile=model_profile,
                            init_local_state = None,
                            local_state_preserve = True,
                            test_adv_method=args.advt_method,
                            test_adv_epsilon=args.advt_epsilon,
                            test_adv_alpha=args.advt_alpha,
                            test_adv_T=args.advt_T,
                            test_adv_norm = args.advt_norm,
                            test_adv_bound = args.advt_bound,
                            device=device,verbose=args.verbose,
                            random_seed=i+args.device_random_seed,
                            reserved_memory=args.reserved_mem
                            ) for i in range(args.num_users)]
            
        

        ## ==================================Build Monitor==================================
        # statistical monitor
        if args.adv_test:
            stat_monitor = AT_Stat_Monitor(clients=clients,weights=weights,
                                           log_path=None)
        else:
            stat_monitor = ST_Stat_Monitor(clients=clients,weights=weights,
                                           log_path=None)

        # systematic monitor
        sys_monitor = Sys_Monitor(clients=clients,log_path=None)
        
        ##  ==================================Build Scheduler==================================
        scheduler = base_AT_scheduler(vars(args))
        
        ## ==================================Build Selector==================================
        
        selector = Random_Selector(total_client_num = args.num_users,
                                       weights = weights)
        
        ## ==================================Build Server==================================
        
        server = Avg_Server(global_model=global_model,
                            clients = clients,
                            selector = selector,
                            scheduler = scheduler,
                            stat_monitor=stat_monitor,
                            sys_monitor=sys_monitor,
                            frac=args.frac,
                            weights=weights,
                            test_dataset=None,
                            device=device,
                            test_every = args.test_every)
        server.load(os.path.join(file_name,'best_model.pt'))
        
        res = server.val(server.global_model)
        
        ## ==================================Final Results==================================
        print("|---- Val Accuracy: {:.2f}%".format(100*res["weighted_val_acc"]))
        
        if args.adv_test:
            print("|---- Val Adv Accuracy: {:.2f}%".format(100*res["weighted_val_adv_acc"]))
            