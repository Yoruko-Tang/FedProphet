#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch

import datasets
from datasets.sampling import iid, SPC_noniid,Dirichlet_noniid
from datasets.language import shakespeare,sent140
from torch.utils.data import Subset

import numpy as np
from numpy.random import RandomState
# from random import Random
import random

import os


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def get_dataset(args,seed=None):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    rs = RandomState(seed)
    assert args.dataset in datasets.__dict__.keys(), "The dataset {} is not supported!".format(args.dataset)
    
    if datasets.dataset_to_modelfamily[args.dataset] in ['mnist','cifar','imagenet']: # CV datasets
        dataset = datasets.__dict__[args.dataset]
        modelfamily = datasets.dataset_to_modelfamily[args.dataset]
        train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
        test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
        train_dataset = dataset(train=True, transform=train_transform,download=True)
        test_dataset = dataset(train=False, transform=test_transform,download=True)
        args.num_classes = len(train_dataset.classes)
    
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = iid(train_dataset, args.num_users,rs)
            user_groups_test = iid(test_dataset,args.num_users,rs)
        else:
            # Sample Non-IID user data from Mnist
            if args.alpha is not None:
                user_groups,_ = Dirichlet_noniid(train_dataset, args.num_users,args.alpha,rs)
                user_groups_test,_ = Dirichlet_noniid(test_dataset, args.num_users,args.alpha,rs)
            else:
                # Chose euqal splits for every user
                user_groups = SPC_noniid(train_dataset, args.num_users,args.shards_per_client,rs)
                user_groups_test = SPC_noniid(test_dataset, args.num_users,args.shards_per_client,rs)

    elif args.dataset == 'shake':
        args.num_classes = 80
        data_dir = './data/shakespeare/'
        user_groups_test={}
        train_dataset,test_dataset,user_groups=shakespeare(data_dir,args.shards_per_client,rs)
    elif args.dataset == 'sent':
        args.num_classes = 2
        data_dir = './data/sent140/'
        user_groups_test={}
        train_dataset,test_dataset,user_groups=sent140(data_dir,args.shards_per_client,rs)
        
    else:
        raise RuntimeError("Not registered dataset! Please register it in utils.py")
    
    args.num_users=len(user_groups.keys())
    weights = []
    for i in range(args.num_users):
        weights.append(len(user_groups[i])/len(train_dataset))
    
    
    return train_dataset, test_dataset, user_groups, user_groups_test,np.array(weights)

def get_data_matrix(dataset,user_groups,num_classes):
    num_users = len(user_groups.keys())
    data_matrix = np.zeros([num_users,num_classes],dtype = np.int64)
    for i in range(num_users):
        subset = Subset(dataset,user_groups[i])
        for _,label in subset:
            data_matrix[i,label] = data_matrix[i,label] + 1
    return data_matrix
        


def average_weights(w,omega=None):
    """
    Returns the average of the weights.
    """
    if omega is None:
        # default : all weights are equal
        omega = np.ones(len(w))
        
    omega = omega/np.sum(omega)
    w_avg = copy.deepcopy(w[0])
    for key in w[0].keys():
        avg_molecule = 0
        for i in range(len(w)):
            avg_molecule+=w[i][key]*omega[i]
        w_avg[key] = copy.deepcopy(avg_molecule)
    return w_avg


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
    
