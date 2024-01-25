#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
# from random import Random
import random
import os


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True




def get_log_path(args):

    if args.sys_efficiency_mode == 'bias' and args.sys_gamma == 1.0 and args.sys_theta == 0:
        child_folder = 'objects'
    if args.sys_efficiency_mode == 'bias' and args.sys_gamma < 1 and args.sys_theta > 0:
        child_folder = 'energy-bias'
    if args.sys_efficiency_mode == 'energy-efficiency':
        child_folder = 'energy-efficiency'

    if not args.iid:
        base_file = './save/{}/{}_{}_{}_{}_N[{}]_C[{}]_iid[{}]_{}[{}]_E[{}]_B[{}]_mu[{}]_lr[{:.5f}]'.\
                    format(child_folder, args.dataset,'FedProx[%.3f]'%args.mu if args.FedProx else 'FedAvg', args.model, args.epochs,args.num_users,args.frac, args.iid,
                    'sp' if args.alpha is None else 'alpha',args.shards_per_client if args.alpha is None else args.alpha,
                    args.local_ep, args.local_bs,args.mu,args.lr)
    else:
        base_file = './save/{}/{}_{}_{}_{}_N[{}]_C[{}]_iid[{}]_E[{}]_B[{}]_mu[{}]_lr[{:.5f}]'.\
                    format(child_folder, args.dataset,'FedProx[%.3f]'%args.mu if args.FedProx else 'FedAvg', args.model, args.epochs,args.num_users,args.frac, args.iid,
                    args.local_ep, args.local_bs,args.mu,args.lr)
    
    
    if args.frac == 1.0:
        file_name = base_file + '/all'
    elif 'fedcor' in args.strategy:
        file_name = base_file + '/gpr[int{}_gam{}_norm{}_disc{}]'.\
            format(args.GPR_interval,args.GPR_gamma,args.poly_norm,args.discount)
        if args.strategy == 'cfedcor':
            file_name += '_cluster[num{}_th{}_rho{}]'.format(args.num_cluster,args.clustering_th,args.rho)
        elif args.rho is not None:
            file_name += '_rho{}'.format(args.rho)
    elif args.strategy == 'harmony':
        file_name = base_file + '/harmony[omega{}]'.format(args.omega)
    else:
        file_name = base_file + '/' + args.strategy
    
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    
    sys_file_name = file_name + '/sys_mode_{}_sys_gamma{}_sys_theta{}_device_seed{}_rho{}'. \
        format(args.sys_efficiency_mode, args.sys_gamma, args.sys_theta, args.random_device_seed, args.rho)

    return file_name, sys_file_name


def exp_details(args):
    print('\nExperimental details:')
    print('    Model     : {}'.format(args.model))
    print('    Optimizer : {}'.format(args.optimizer))
    print('    Learning  : {}'.format(args.lr))
    print('    Global Rounds   : {}'.format(args.epochs))

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print('    Fraction of users  : {}'.format(args.frac))
    print('    Local Batch size   : {}'.format(args.local_bs))
    print('    Local Epochs       : {}\n'.format(args.local_ep))
    if args.FedProx:
        print('    Algorithm    :    FedProx({})'.format(args.mu))
    else:
        print('    Algorithm    : FedAvg')
    print('    Selection Strategy    : {}'.format(args.strategy))
    return

if __name__ == "__main__":
    from options import args_parser
    import matplotlib.pyplot as plt
    ALL_LETTERS = np.array(list("\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"))
    args = args_parser()
    args.dataset = 'sent'
    args.shards_per_client=1
    print(args.dataset)
    train_dataset, test_dataset, user_groups, user_groups_test,weights = get_dataset(args)
    print(len(train_dataset))
    print(len(test_dataset))
    # print(train_dataset[100][0].max())
    # print(''.join(ALL_LETTERS[train_dataset[0][0].numpy()].tolist()))
    # print(''.join(ALL_LETTERS[train_dataset[0][1].numpy()].tolist()))
    print(args.num_users)
    plt.hist(weights,bins=20)
    plt.show()
    
