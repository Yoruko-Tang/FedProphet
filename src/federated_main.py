import os.path as osp
import time
import datetime
import json
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from copy import deepcopy

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
    # save the running options
    with open(osp.join(base_file,'params.json'),'w') as pf:
        json.dump(params,pf,indent=True)
    
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

        if args.flalg in ["FedDF","FedET"]:
            subset_idxs = np.random.choice(range(len(test_dataset)),args.public_dataset_size,replace=False)
            public_dataset = Subset(test_dataset,subset_idxs)

        ## ==================================Build Model==================================
        # try to get the model from torchvision.models with pretraining
        global_model = get_net(modelname = args.model_arch,
                               modeltype = dataset_to_datafamily[args.dataset],
                               num_classes = args.num_classes,
                               pretrained = args.pretrained,
                               norm_type = args.norm,
                               adv_norm = (args.adv_train or args.adv_test),
                               modularization = (args.flalg == "FedProphet"))
        print(global_model)
        model_profile = model_summary(model = deepcopy(global_model),
                                      inputsize=[args.local_bs]+list(train_dataset[0][0].shape),
                                      default_local_eps=args.local_ep,
                                      **vars(args))
        
        if args.flalg in ['FedET','FedDF']:# generate small edge models
            edge_models = []
            for m in args.edge_model_archs:
                edge_models.append(get_net(modelname = m,
                                    modeltype = dataset_to_datafamily[args.dataset],
                                    num_classes = args.num_classes,
                                    pretrained = args.pretrained,
                                    norm_type = args.norm,
                                    adv_norm = (args.adv_train or args.adv_test)))
            print(edge_models)
            edge_model_profiles = []
            for mod in edge_models:
                edge_model_profiles.append(model_summary(model = deepcopy(mod),
                                            inputsize=[args.local_bs]+list(train_dataset[0][0].shape),
                                            default_local_eps=args.local_ep,
                                            **vars(args)))

        ## ==================================Build Clients==================================
        if args.flalg in ['FedAvg']:
            clients = [AT_Client(train_dataset,user_groups[i],
                                sys_info=user_devices[i],
                                model_profile=model_profile,
                                local_state_preserve = False,
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
                  
        elif args.flalg in ['FedBN']:
            clients = [AT_Client(train_dataset,user_groups[i],
                                sys_info=user_devices[i],
                                model_profile=model_profile,
                                init_local_state = ST_Client.get_local_state_dict(global_model),
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
            
        elif args.flalg == 'FedProphet':
            clients = [Module_Client(train_dataset,user_groups[i],
                                     sys_info=user_devices[i],
                                     model_profile=model_profile,
                                     init_local_state = ST_Client.get_local_state_dict(global_model),
                                     local_state_preserve = (args.norm == 'BN'),
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
        
        elif args.flalg in ["FedDF","FedET"]:
            clients = [Multimodel_Client(train_dataset,user_groups[i],
                                sys_info=user_devices[i],
                                model_profile=edge_model_profiles[0],
                                init_local_state = [ST_Client.get_local_state_dict(mod) for mod in edge_models],
                                local_state_preserve = (args.norm == 'BN'),
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
        # Todo: add other types of clients
        else:
            raise RuntimeError("Not supported FL optimizer: "+args.flalg) 

        ## ==================================Build Monitor==================================
        # statistical monitor
        if args.adv_test:
            stat_monitor = AT_Stat_Monitor(clients=clients,weights=weights,
                                           log_path=file_name)
        else:
            stat_monitor = ST_Stat_Monitor(clients=clients,weights=weights,
                                           log_path=file_name)

        # systematic monitor
        sys_monitor = Sys_Monitor(clients=clients,log_path=file_name)
        
        ##  ==================================Build Scheduler==================================
        if args.flalg in ["FedAvg","FedBN"]:
            scheduler = base_AT_scheduler(vars(args))
        
        elif args.flalg == "FedProphet":
            scheduler = module_scheduler(vars(args),
                                         model_profile=model_profile,
                                         clients=clients,
                                         log_path=file_name)
        elif args.flalg in ["FedET","FedDF"]:
            scheduler = kd_scheduler(vars(args),
                                     model_profiles=edge_model_profiles,
                                     global_val = (args.flalg == "FedET"))
        # Todo: Add schedulers for other baselines
        else:
            raise RuntimeError("FL optimizer {} has no registered scheduler!".format(args.flalg)) 
            
        ## ==================================Build Selector==================================
        if args.strategy == 'rand':
            selector = Random_Selector(total_client_num = args.num_users,
                                       weights = weights)
        elif args.strategy == 'afl':
            selector = AFL_Selector(total_client_num = args.num_users,
                                    weights = weights,**params)
        elif args.strategy == 'powerd':
            selector = PowerD_Selector(total_client_num = args.num_users,
                                       weights = weights,**params)
        elif args.strategy == 'oort':
            selector = Oort_Selector(total_client_num = args.num_users,
                                     weights = weights,**params) 
        elif args.strategy == 'fedcbs':
            selector = FedCBS_Selector(total_client_num = args.num_users,
                                       weights = weights,
                                       data_matrix = data_matrix,**params)
        elif args.strategy == 'harmony':
            selector = Harmony_Selector(total_client_num = args.num_users,
                                        weights = weights,
                                        data_matrix = data_matrix,**params)
        elif args.strategy == 'fedcor':
            selector = FedCor_Selector(total_client_num = args.num_users,
                                       weights = weights,
                                       clustered = False,**params)
        elif args.strategy == 'fedrepre':
            selector = FedCor_Selector(total_client_num = args.num_users,
                                       weights = weights,
                                       clustered = True,**params)
        else:
            raise NotImplementedError("Not a supported selection strategy: {}".format(args.strategy))
        
        ## ==================================Build Server==================================
        if args.flalg in ["FedAvg"]:
            server = Avg_Server(global_model=global_model,
                                clients = clients,
                                selector = selector,
                                scheduler = scheduler,
                                stat_monitor=stat_monitor,
                                sys_monitor=sys_monitor,
                                frac=args.frac,
                                weights=weights,
                                test_dataset=test_dataset,
                                local_state_preserve=False,
                                device=device,
                                test_every = args.test_every)
        elif args.flalg in ["FedBN"]:
            server = Avg_Server(global_model=global_model,
                                clients = clients,
                                selector = selector,
                                scheduler = scheduler,
                                stat_monitor=stat_monitor,
                                sys_monitor=sys_monitor,
                                frac=args.frac,
                                weights=weights,
                                device=device,
                                test_every = args.test_every)
        elif args.flalg == "FedProphet":
            server = Fedprophet_Avg_Server(global_model=global_model,
                                           clients = clients,
                                           selector = selector,
                                           scheduler = scheduler,
                                           stat_monitor=stat_monitor,
                                           sys_monitor=sys_monitor,
                                           frac=args.frac,
                                           weights=weights,
                                           test_dataset=None if args.norm == 'BN' else test_dataset,
                                           local_state_preserve=(args.norm == 'BN'),
                                           device=device,
                                           test_every = args.test_every)
        elif args.flalg == "FedDF":
            server = FedDF_Server(global_model=global_model,
                                edge_models = edge_models,
                                clients = clients,
                                selector = selector,
                                scheduler = scheduler,
                                stat_monitor=stat_monitor,
                                sys_monitor=sys_monitor,
                                frac=args.frac,
                                weights=weights,
                                test_dataset=test_dataset,
                                local_state_preserve=False,
                                device=device,
                                test_every = args.test_every,
                                public_dataset = public_dataset,
                                dist_iters = args.dist_iters,
                                dist_lr = args.dist_lr,
                                dist_batch_size = args.dist_batch_size)
        elif args.flalg == "FedET":
            server = FedET_Server(global_model=global_model,
                                edge_models = edge_models,
                                clients = clients,
                                selector = selector,
                                scheduler = scheduler,
                                stat_monitor=stat_monitor,
                                sys_monitor=sys_monitor,
                                frac=args.frac,
                                weights=weights,
                                test_dataset=test_dataset,
                                local_state_preserve=False,
                                device=device,
                                test_every = args.test_every,
                                public_dataset = public_dataset,
                                dist_iters = args.dist_iters,
                                dist_lr = args.dist_lr,
                                dist_batch_size = args.dist_batch_size,
                                diver_lamb = args.diver_lamb)

        server.monitor() # initialize the stat_info and sys_info at the beginning
        
        ## ==================================Start Training==================================
        for epoch in tqdm(range(args.epochs)):
            CTN = server.train()
            if not CTN:
                break      

        
        ## ==================================Final Results==================================
        print(' \n Results after {} global rounds of training:'.format(epoch+1))
        print("|---- Final Val Accuracy: {:.2f}%".format(100*stat_monitor.weighted_global_accs[-1]))
        print("|---- Max Val Accuracy: {:.2f}%".format(100*max(stat_monitor.weighted_global_accs)))
        print("|---- Final Test Accuracy: {:.2f}%".format(100*stat_monitor.test_accs[-1]))
        print("|---- Max Test Accuracy: {:.2f}%".format(100*max(stat_monitor.test_accs)))
        
        if args.adv_test:
            print("|---- Final Val Adv Accuracy: {:.2f}%".format(100*stat_monitor.weighted_global_adv_accs[-1]))
            print("|---- Max Val Adv Accuracy: {:.2f}%".format(100*max(stat_monitor.weighted_global_adv_accs)))
            print("|---- Final Test Adv Accuracy: {:.2f}%".format(100*stat_monitor.test_adv_accs[-1]))
            print("|---- Max Test Adv Accuracy: {:.2f}%".format(100*max(stat_monitor.test_adv_accs)))
            

        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

                