import os
import os.path as osp
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import random
import torch

from client import *
from server import *
from selection import *
from monitor import *
from scheduler import *


from utils.options import args_parser
from utils.models import get_net
from utils.utils import setup_seed, get_log_path, exp_details
from utils.runtime import sys_efficiency_U



from datasets.dataset_utils import get_dataset, get_data_matrix
from datasets import dataset_to_modelfamily






if __name__ == '__main__':
    os.environ["OUTDATED_IGNORE"]='1'
    start_time = time.time()
    # path_project = os.path.abspath('..')
    args = args_parser()
    gargs = copy.deepcopy(args)
    params = vars(gargs)
    exp_details(args)
    file_name, sys_file_name = get_log_path(args)
    
    device = torch.device('cuda:'+args.gpu if args.gpu else 'cpu')
    
    if args.gpu:
        torch.cuda.set_device(device)

    for seed in args.seed: ## fix seed
        args = copy.deepcopy(gargs)# recover the args
        print("Start with Random Seed: {}".format(seed))
        
        ## ==================================Load Dataset==================================
        train_dataset, test_dataset, user_groups, user_groups_test, weights = get_dataset(args,seed)
        data_matrix = get_data_matrix(train_dataset, user_groups, args.num_classes)
        if seed is not None:
            setup_seed(seed)

        ## ==================================Build Model==================================
        # try to get the model from torchvision.models with pretraining
        global_model = get_net(modelname = args.model,
                               modeltype=dataset_to_modelfamily[args.dataset],
                               pretrained = args.pretrained,
                               num_classes = args.num_classes)
        print(global_model)


        ## ==================================Build Clients==================================
        if args.flalg == 'FedAvg':
            clients = [ST_Client(train_dataset,user_groups[i],local_state_preserve=False,device=device) for i in range(args.num_users)]
        elif args.flalg == 'FedBN':
            clients = [ST_Client(train_dataset,user_groups[i],local_state_preserve=True,device=device) for i in range(args.num_users)]
        # elif args.flalg == 'FAT':
        #     clients = [AT_Client(train_dataset,user_groups[i],local_state_preserve=False,device=device) for i in range(args.num_users)]
        # Todo: add other types of clients
            

        ## ==================================Build Monitor==================================
        if args.adv_test:
            stat_monitor = AT_Stat_Monitor(clients=clients,weights=weights,log_path = osp.join(file_name,str(seed)))
        else:
            stat_monitor = ST_Stat_Monitor(clients=clients,weights=weights,log_path = osp.join(file_name,str(seed)))

        # Todo: Replace with sys monitor
        sys_monitor=None
        # system information
        training_latency_per_epoch = []
        training_energy_per_epoch = []
        device_id_per_epoch = []
        degrade_factor_per_epoch = []
        true_degrade_factor_per_epoch = []
        runtime_app_id_per_epoch = []
        network_bw_per_epoch = []
        network_latency_per_epoch = []
        Ptx_per_epoch = []
        network_id_per_epoch = []


        # Estimate the global system before training
        random.seed(seed)
        runtime_seed_per_epoch = [random.randint(1000, 100000) for ep in range(args.epochs)]
        cost_per_client_list, cost_true_per_client_list, training_latency_per_client, training_energy_per_client, device_id_list, degrade_factor_per_client_list, \
        true_degrade_factor_per_client_list, runtime_app_id_list, network_bw_per_client_list, network_latency_per_client_list, Ptx_per_client_list, network_id_list \
        = sys_efficiency_U(args.sys_efficiency_mode, args.sys_gamma, args.sys_theta, args.flsys_info, args.model, dataset_to_modelfamily[args.dataset], args.num_users, args.random_device_seed, runtime_seed_per_epoch[0])

        
        ##  ==================================Build Scheduler==================================
        if args.flalg in ["FedAvg","FedBN","FedAvgAT","FedBNAT"]:
            scheduler = base_scheduler(**args)
            
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
        if args.flalg in ["FedAvg","FedBN","FedAvgAT","FedBNAT"]:
            server = Avg_Server(global_model=global_model,clients = clients,
                                selector = selector,scheduler = scheduler,
                                stat_monitor=stat_monitor,sys_monitor=sys_monitor,
                                weights=weights,test_dataset=test_dataset,
                                device=device,test_every = args.test_every)


        ## ==================================Start Training==================================
        for epoch in tqdm(range(args.epochs)):
            
            # ## =============================== system information collecion ===============================
            # if epoch > 0:
            #     cost_per_client_list, cost_true_per_client_list, training_latency_per_client, training_energy_per_client, device_id_list, degrade_factor_per_client_list, \
            #     true_degrade_factor_per_client_list, runtime_app_id_list, network_bw_per_client_list, network_latency_per_client_list, Ptx_per_client_list, network_id_list \
            #     = sys_efficiency_U(args.sys_efficiency_mode, args.sys_gamma, args.sys_theta, args.flsys_info, args.model, dataset_to_modelfamily[args.dataset], args.num_users, args.random_device_seed, runtime_seed_per_epoch[epoch])

            # # system information of selected clients
            # training_latency_per_epoch.append([training_latency_per_client[idx] for idx in idxs_users])
            # training_energy_per_epoch.append([training_energy_per_client[idx] for idx in idxs_users])
            # device_id_per_epoch.append([device_id_list[idx] for idx in idxs_users])
            # degrade_factor_per_epoch.append([degrade_factor_per_client_list[idx] for idx in idxs_users])
            # true_degrade_factor_per_epoch.append([true_degrade_factor_per_client_list[idx] for idx in idxs_users])
            # runtime_app_id_per_epoch.append([runtime_app_id_list[idx] for idx in idxs_users])
            # network_bw_per_epoch.append([network_bw_per_client_list[idx] for idx in idxs_users])
            # network_latency_per_epoch.append([network_latency_per_client_list[idx] for idx in idxs_users])
            # Ptx_per_epoch.append([Ptx_per_client_list[idx] for idx in idxs_users])
            # network_id_per_epoch.append([network_id_list[idx] for idx in idxs_users])

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

        
        
        # # save the system info during training:
        # with open(sys_file_name+'_{}.pkl'.format(seed), 'wb') as sys_f:
        #     pickle.dump([training_latency_per_epoch, training_energy_per_epoch, device_id_per_epoch, degrade_factor_per_epoch, true_degrade_factor_per_epoch, \
        #                  runtime_app_id_per_epoch, network_bw_per_epoch, network_latency_per_epoch, network_id_per_epoch, Ptx_per_epoch], sys_f)
        
        
                