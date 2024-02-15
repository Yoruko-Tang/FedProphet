import os.path as osp
import time
import datetime
import json
from tqdm import tqdm
import torch
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

        ## ==================================Build Model==================================
        # try to get the model from torchvision.models with pretraining
        global_model = get_net(modelname = args.model,
                               modeltype = dataset_to_datafamily[args.dataset],
                               num_classes = args.num_classes,
                               pretrained = args.pretrained,
                               adv_norm = (args.adv_train or args.adv_test),
                               modularization = (args.flalg == "FedProphet"))
        print(global_model)
        model_profile = model_summary(model = deepcopy(global_model),
                                      inputsize=[args.local_bs]+list(train_dataset[0][0].shape),
                                      default_local_eps=args.local_ep*(args.adv_T+1 if args.adv_train else 1))
        

        ## ==================================Build Clients==================================
        if args.flalg == 'FedAvg':
            clients = [ST_Client(train_dataset,user_groups[i],
                                 sys_info=user_devices[i],
                                 model_profile=model_profile,
                                 local_state_preserve = False,
                                 device=device,verbose=args.verbose,
                                 random_seed=i+args.device_random_seed
                                 ) for i in range(args.num_users)]
        elif args.flalg == 'FedBN':
            clients = [ST_Client(train_dataset,user_groups[i],
                                 sys_info=user_devices[i],
                                 model_profile=model_profile,
                                 local_state_preserve = True,
                                 device=device,verbose=args.verbose,
                                 random_seed=i+args.device_random_seed
                                 ) for i in range(args.num_users)]
        
        elif args.flalg == 'FedAvgAT':
            clients = [AT_Client(train_dataset,user_groups[i],
                                 sys_info=user_devices[i],
                                 model_profile=model_profile,
                                 local_state_preserve = False,
                                 test_adv_method=args.advt_method,
                                 test_adv_epsilon=args.advt_epsilon,
                                 test_adv_alpha=args.advt_alpha,
                                 test_adv_T=args.advt_T,
                                 device=device,verbose=args.verbose,
                                 random_seed=i+args.device_random_seed
                                 ) for i in range(args.num_users)]
        elif args.flalg == 'FedBNAT':
            clients = [AT_Client(train_dataset,user_groups[i],
                                 sys_info=user_devices[i],
                                 model_profile=model_profile,
                                 local_state_preserve = True,
                                 test_adv_method=args.advt_method,
                                 test_adv_epsilon=args.advt_epsilon,
                                 test_adv_alpha=args.advt_alpha,
                                 test_adv_T=args.advt_T,
                                 device=device,verbose=args.verbose,
                                 random_seed=i+args.device_random_seed
                                 ) for i in range(args.num_users)]

        # Todo: add other types of clients
        else:
            raise RuntimeError("Not supported FL optimizer: "+args.flalg) 

        ## ==================================Build Monitor==================================
        # statistical monitor
        if args.adv_test:
            stat_monitor = AT_Stat_Monitor(clients=clients,weights=weights,
                                           log_path = file_name,
                                           test_adv_method=args.advt_method,
                                           test_adv_eps=args.advt_epsilon,
                                           test_adv_alpha=args.advt_alpha,
                                           test_adv_T=args.advt_T,
                                           test_adv_norm=args.advt_norm,
                                           test_adv_bound=args.advt_bound)
        else:
            stat_monitor = ST_Stat_Monitor(clients=clients,weights=weights,
                                           log_path = file_name)

        # systematic monitor
        sys_monitor = Sys_Monitor(clients=clients,log_path=file_name)
        
        ##  ==================================Build Scheduler==================================
        if args.flalg in ["FedAvg","FedBN"]:
            scheduler = base_scheduler(vars(args))
        elif args.flalg in ["FedAvgAT","FedBNAT"]:
            scheduler = base_AT_scheduler(vars(args))
        elif args.flalg == "FedProphet":
            scheduler = module_scheduler(vars(args),model_profile)
        # Todo: Add schedulers for other baselines
        else:
            raise RuntimeError("FL optimizer {} has no registered scheduler!".format(args.flalg)) 
            
        ## ==================================Build Selector==================================
        if args.strategy == 'rand':
            selector = Random_Selector(total_client_num = args.num_users,weights = weights)
        elif args.strategy == 'afl':
            selector = AFL_Selector(total_client_num = args.num_users,weights = weights,**params)
        elif args.strategy == 'powerd':
            selector = PowerD_Selector(total_client_num = args.num_users,weights = weights,**params)
        elif args.strategy == 'oort':
            # set gamma = 1.0 for only time in cost
            selector = Oort_Selector(total_client_num = args.num_users,weights = weights,**params) 
        elif args.strategy == 'fedcbs':
            selector = FedCBS_Selector(total_client_num = args.num_users,weights = weights,data_matrix = data_matrix,**params)
        elif args.strategy == 'harmony':
            # set gamma = 1.0 for only time in cost
            selector = Harmony_Selector(total_client_num = args.num_users,weights = weights,data_matrix = data_matrix,**params)
        elif args.strategy == 'fedcor':
            selector = FedCor_Selector(total_client_num = args.num_users,weights = weights,clustered = False,**params)
        elif args.strategy == 'cfedcor':
            selector = FedCor_Selector(total_client_num = args.num_users,weights = weights,clustered = True,**params)
        else:
            raise NotImplementedError("Not a supported selection strategy: {}".format(args.strategy))
        
        ## ==================================Build Server==================================
        if args.flalg in ["FedAvg","FedAvgAT"]:
            server = Avg_Server(global_model=global_model,clients = clients,
                                selector = selector,scheduler = scheduler,
                                stat_monitor=stat_monitor,sys_monitor=sys_monitor,
                                frac=args.frac,weights=weights,test_dataset=test_dataset,
                                device=device,test_every = args.test_every)
        if args.flalg in ["FedBN","FedBNAT"]:
            server = Avg_Server(global_model=global_model,clients = clients,
                                selector = selector,scheduler = scheduler,
                                stat_monitor=stat_monitor,sys_monitor=sys_monitor,
                                frac=args.frac,weights=weights,test_dataset=None,
                                device=device,test_every = args.test_every)


        ## ==================================Start Training==================================
        for epoch in tqdm(range(args.epochs)):
            server.train()
                          

        
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

                