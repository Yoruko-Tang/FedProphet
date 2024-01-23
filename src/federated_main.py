#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import random
import torch



from utils.options import args_parser
from utils.update import LocalUpdate,test_inference,train_federated_learning,federated_test_idx
from utils.models import MLP, NaiveCNN, BNCNN, RNN, LeNet5
from utils.models import get_net
from utils.utils import get_dataset, get_data_matrix, average_weights, exp_details,setup_seed, get_log_path
from utils.runtime import sys_efficiency_U
from utils.mvnt import MVN_Test

from selection import *
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
    
    device = 'cuda:'+args.gpu if args.gpu else 'cpu'
    
    if args.gpu:
        torch.cuda.set_device(device)

    if gargs.seed is None or gargs.iid:
        gargs.seed = [None,]
    for seed in gargs.seed: ## fix seed
        args = copy.deepcopy(gargs)# recover the args
        print("Start with Random Seed: {}".format(seed))
        
        ## ==================================Load Dataset==================================
        train_dataset, test_dataset, user_groups, user_groups_test, weights = get_dataset(args,seed)
        data_matrix = get_data_matrix(train_dataset, user_groups, args.num_classes)
        if seed is not None:
            setup_seed(seed)
        data_size = train_dataset[0][0].shape

        ## ==================================Build Model==================================
        if args.model == 'cnn':
            # Naive Convolutional neural netork
            global_model = NaiveCNN(args=args,input_shape = data_size,final_pool=False)
        
        elif args.model == 'LeNet5':
            global_model = LeNet5()
        
        elif args.model == 'bncnn':
            # Convolutional neural network with batch normalization
            global_model = BNCNN(args = args, input_shape = data_size)

        elif args.model == 'mlp' or args.model == 'log':
            # Multi-layer preceptron
            len_in = 1
            for x in data_size:
                len_in *= x
                global_model = MLP(dim_in=len_in, dim_hidden=args.mlp_layers if args.model=='mlp' else [],
                                dim_out=args.num_classes)
        elif args.model == 'rnn':
            if args.dataset=='shake':
                global_model = RNN(256,args.num_classes)
            else:
                # emb_arr,_,_= get_word_emb_arr('./data/sent140/embs.json')
                global_model = RNN(256,args.num_classes,300,True,128)
        else: # try to get the model from torchvision.models with pretraining
            global_model = get_net(modelname = args.model,modeltype=dataset_to_modelfamily[args.dataset],
                                   pretrained = args.pretrained,num_classes = args.num_classes)


        print(global_model)

        # copy weights
        global_weights = global_model.state_dict()
        local_weights = []# store local weights of all users for averaging
        local_states = []# store local states of all users, these parameters should not be uploaded

        
        for i in range(args.num_users): # store the local models on cpu for less gpu memory requirement
            local_states.append(copy.deepcopy(global_model.Get_Local_State_Dict()))
            local_weights.append(copy.deepcopy(global_weights))

        local_states = np.array(local_states)
        local_weights = np.array(local_weights)


        ## ==================================Define Statistics==================================
        train_loss, train_accuracy = [], []
        test_losses,test_accuracy = [],[]
        max_accuracy=0.0

        local_losses = []# test losses evaluated on local models(before averaging)
        chosen_clients = []# chosen clients on each epoch
        gt_global_losses = []# test losses on global models(after averaging) over all clients
        
        print_every = 1
        init_mu = args.mu

        predict_losses = []# GPR prediction loss
        embeddings = []
        centroids = []

        sigma = []
        sigma_gt=[]

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

        # Test the global model before training
        list_acc, list_loss = federated_test_idx(args,global_model,
                                                list(range(args.num_users)),
                                                train_dataset,user_groups)
        gt_global_losses.append(list_loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        loss_prev = sum(list_loss)/len(list_loss)

        # Estimate the global system before training
        random.seed(seed)
        runtime_seed_per_epoch = [random.randint(1000, 100000) for ep in range(args.epochs)]
        cost_per_client_list, cost_true_per_client_list, training_latency_per_client, training_energy_per_client, device_id_list, degrade_factor_per_client_list, \
        true_degrade_factor_per_client_list, runtime_app_id_list, network_bw_per_client_list, network_latency_per_client_list, Ptx_per_client_list, network_id_list \
        = sys_efficiency_U(args.sys_efficiency_mode, args.sys_gamma, args.sys_theta, args.flsys_info, args.model, dataset_to_modelfamily[args.dataset], args.num_users, args.random_device_seed, runtime_seed_per_epoch[0])

        ## ==================================Build Selector==================================
        if args.strategy == 'rand':
            selector = Random_Selector(total_client_num = args.num_users,weights = weights)
        elif args.strategy == 'afl':
            selector = AFL_Selector(total_client_num = args.num_users,weights = weights,loss_value = list_loss,**params)
        elif args.strategy == 'powerd':
            selector = PowerD_Selector(total_client_num = args.num_users,weights = weights,loss_value = list_loss,**params)
        elif args.strategy == 'oort':
            # set gamma = 1.0 for only time in cost
            selector = Oort_Selector(total_client_num = args.num_users,weights = weights,loss_init = list_loss, sys_init = cost_per_client_list,**params) 
        elif args.strategy == 'fedcbs':
            selector = FedCBS_Selector(total_client_num = args.num_users,weights = weights,data_matrix = data_matrix,**params)
        elif args.strategy == 'harmony':
            # set gamma = 1.0 for only time in cost
            selector = Harmony_Selector(total_client_num = args.num_users,weights = weights,data_matrix = data_matrix, runtime = cost_per_client_list,**params)
        elif args.strategy == 'fedcor':
            selector = FedCor_Selector(total_client_num = args.num_users,weights = weights,clustered = False,**params)
        elif args.strategy == 'cfedcor':
            selector = FedCor_Selector(total_client_num = args.num_users,weights = weights,clustered = True,**params)
        
        else:
            raise NotImplementedError("Not a supported selection strategy: {}".format(args.strategy))

        ## ==================================Start Training==================================
        for epoch in tqdm(range(args.epochs)):
            print('\n | Global Training Round : {} |\n'.format(epoch+1))
            epoch_global_losses = []
            epoch_local_losses = []
            global_model.train()
            if args.dataset=='cifar' or epoch in args.schedule:
                args.lr*=args.lr_decay
                   
            m = max(int(args.frac * args.num_users), 1)

            idxs_users = selector.select(m)
            
            print("Chosen Clients:",idxs_users)

            chosen_clients.append(idxs_users)

            ## =============================== system information collecion ===============================
            if epoch > 0:
                cost_per_client_list, cost_true_per_client_list, training_latency_per_client, training_energy_per_client, device_id_list, degrade_factor_per_client_list, \
                true_degrade_factor_per_client_list, runtime_app_id_list, network_bw_per_client_list, network_latency_per_client_list, Ptx_per_client_list, network_id_list \
                = sys_efficiency_U(args.sys_efficiency_mode, args.sys_gamma, args.sys_theta, args.flsys_info, args.model, dataset_to_modelfamily[args.dataset], args.num_users, args.random_device_seed, runtime_seed_per_epoch[epoch])

            # system information of selected clients
            training_latency_per_epoch.append([training_latency_per_client[idx] for idx in idxs_users])
            training_energy_per_epoch.append([training_energy_per_client[idx] for idx in idxs_users])
            device_id_per_epoch.append([device_id_list[idx] for idx in idxs_users])
            degrade_factor_per_epoch.append([degrade_factor_per_client_list[idx] for idx in idxs_users])
            true_degrade_factor_per_epoch.append([true_degrade_factor_per_client_list[idx] for idx in idxs_users])
            runtime_app_id_per_epoch.append([runtime_app_id_list[idx] for idx in idxs_users])
            network_bw_per_epoch.append([network_bw_per_client_list[idx] for idx in idxs_users])
            network_latency_per_epoch.append([network_latency_per_client_list[idx] for idx in idxs_users])
            Ptx_per_epoch.append([Ptx_per_client_list[idx] for idx in idxs_users])
            network_id_per_epoch.append([network_id_list[idx] for idx in idxs_users])

            ## ==================================Local Training==================================
            for idx in idxs_users:
                local_model = copy.deepcopy(global_model)
                local_update = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[idx] ,global_round = epoch)
                local_model,local_test_loss,init_local_test_loss = local_update.update_weights(model=local_model)
                local_model.to('cpu') # store the local models on cpu for less gpu memory consumption
                local_states[idx] = copy.deepcopy(local_model.Get_Local_State_Dict())
                local_weights[idx] = copy.deepcopy(local_model.state_dict())
                epoch_global_losses.append(init_local_test_loss)# TAKE CARE: this is the test loss evaluated on the (t-1)-th global weights!
                epoch_local_losses.append(local_test_loss)


            ## ==================================Server Aggregation==================================
            if args.global_average:
                global_weights = average_weights(local_weights,omega=None)
            else:
                global_weights = average_weights(local_weights[idxs_users],omega=None)

            for i in range(args.num_users):
                local_weights[i] = copy.deepcopy(global_weights)
            
            global_model.load_state_dict(global_weights)

            ## ==================================Collecting Losses==================================
            local_losses.append(epoch_local_losses)
            loss_avg = sum(epoch_local_losses) / len(epoch_local_losses)
            train_loss.append(loss_avg)

            # calculate test accuracy over all users
            list_acc, list_loss = federated_test_idx(args,global_model,
                                                    list(range(args.num_users)),
                                                    train_dataset,user_groups)
            gt_global_losses.append(list_loss)
            train_accuracy.append(sum(list_acc)/len(list_acc))

            # test inference on the global test dataset
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            test_accuracy.append(test_acc)
            test_losses.append(test_loss)
            if args.target_accuracy is not None:
                if test_acc>=args.target_accuracy:
                    break

            # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
                print('Training Loss : {}'.format(np.sum(np.array(list_loss)*weights)))
                print("Test Loss: {}".format(test_loss))
                print("Test Accuracy: {:.2f}%".format(100*test_acc))


            ## ==================================FedProx Adaptation==================================
            if args.dynamic_mu and epoch>0:
                if loss_avg>loss_prev:
                    args.mu+=init_mu*0.1
                else:
                    args.mu=max([args.mu-init_mu*0.1,0.0])
            loss_prev = loss_avg
                       
            
            ## ==================================Update Selector Status==================================
            if args.strategy == 'rand':
                stat_info = None
                sys_info = None
            elif args.strategy == 'afl':
                stat_info = epoch_global_losses # the global loss of selected clients in the last round
                sys_info = None
            elif args.strategy == 'powerd':
                stat_info = gt_global_losses[-1] # the global loss of all clients in this round
                sys_info = None
            elif args.strategy == 'oort':
                stat_info = epoch_local_losses # the local loss of selected clients in this round
                cost_true_selected_clients = [cost_true_per_client_list[idx] for idx in idxs_users]
                sys_info = cost_true_selected_clients # true training + communication time of selected clients in this round
            elif args.strategy == 'fedcbs':
                stat_info = data_matrix # data matrix of all clients
                sys_info = None
            elif args.strategy == 'harmony':
                stat_info = data_matrix # data matrix of all clients
                sys_info = cost_per_client_list # estimated runtime of all clients
            elif 'fedcor' in args.strategy:
                loss_changes = None
                if epoch<=args.warmup:# warm-up
                    loss_changes = np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2])
                elif epoch%args.GPR_interval==0:
                    print("Training with Random Selection For GPR Training:")
                    random_idxs_users = np.random.choice(range(args.num_users), m, replace=False)
                    gpr_acc,gpr_loss = train_federated_learning(args,epoch,
                                        copy.deepcopy(global_model),random_idxs_users,train_dataset,user_groups)
                    loss_changes = np.array(gpr_loss)-np.array(gt_global_losses[-1])
                stat_info = loss_changes # global loss changes of all clients
                sys_info = cost_per_client_list # estimated runtime of all clients
            
            selector.stat_update(epoch=epoch,selected_clients=idxs_users,stat_info=stat_info,sys_info=sys_info)

            
            ## ==================================Test GP==================================
            if 'fedcor' in args.strategy and epoch>args.warmup:
                test_idx = np.random.choice(range(args.num_users), m, replace=False)
                test_data = np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                            np.expand_dims(np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]),1),
                                            np.ones([args.num_users,1])],1)
                pred_idx = np.delete(list(range(args.num_users)),test_idx)
                
                try:
                    predict_loss,mu_p,sigma_p = selector.gpr.Predict_Loss(test_data,test_idx,pred_idx)
                    print("GPR Predict relative Loss:{:.4f}".format(predict_loss))
                    predict_losses.append(predict_loss)
                except:
                    print("[Warning]: Singular posterior covariance encountered, skip the GPR test in this round!")

                if args.strategy == 'cfedcor' and epoch%args.GPR_interval==0:
                    embeddings.append(selector.gpr.Projection(list(range(args.num_users))).clone().detach().cpu().numpy())
                    centroids.append(selector.gpr.centroids.clone().detach().cpu().numpy())

                if args.mvnt and (epoch==args.warmup or (epoch%args.mvnt_interval==0 and epoch>args.warmup)):
                    mvn_file = file_name+'/MVN/{}'.format(seed)
                    if not os.path.exists(mvn_file):
                        os.makedirs(mvn_file)
                    mvn_samples=MVN_Test(args,copy.deepcopy(global_model),
                                                train_dataset,user_groups,
                                                file_name+'/MVN/{}/{}.csv'.format(seed,epoch))
                    sigma_gt.append(np.cov(mvn_samples,rowvar=False,bias = True))
                    sigma.append(selector.gpr.Covariance().clone().detach().numpy())
                
                                    

        
        ## ==================================Final Results==================================
        print(' \n Results after {} global rounds of training:'.format(epoch+1))
        print("|---- Final Test Accuracy: {:.2f}%".format(100*test_accuracy[-1]))
        print("|---- Max Test Accuracy: {:.2f}%".format(100*max(test_accuracy)))
        if 'fedcor' in args.strategy:
            print("|---- Mean GP Prediction Loss: {:.4f}".format(np.mean(predict_losses)))

        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

        # save the training records:
        with open(file_name+'/stat_{}.pkl'.format(seed), 'wb') as f:
            pickle.dump([train_loss, train_accuracy,chosen_clients,
                        weights,selector.gpr.state_dict() if 'fedcor' in args.strategy else None,
                        gt_global_losses,test_losses,test_accuracy], f)
        
        # save the system info during training:
        with open(sys_file_name+'_{}.pkl'.format(seed), 'wb') as sys_f:
            pickle.dump([training_latency_per_epoch, training_energy_per_epoch, device_id_per_epoch, degrade_factor_per_epoch, true_degrade_factor_per_epoch, \
                         runtime_app_id_per_epoch, network_bw_per_epoch, network_latency_per_epoch, network_id_per_epoch, Ptx_per_epoch], sys_f)
        
        if args.strategy == 'cfedcor':
            with open(file_name+'/cfedcor_centroids_{}.pkl'.format(seed), 'wb') as f:
                pickle.dump([embeddings,centroids],f)

        if args.mvnt:
            with open(file_name+'/MVNSigma_{}.pkl'.format(seed), 'wb') as f:
                pickle.dump([sigma,sigma_gt],f)

                