#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import math

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="total number of clients: N")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--schedule', type=int, nargs='*', default=[162, 244],
                        help='Decrease learning rate at these rounds.')
    parser.add_argument('--lr_decay',type = float,default=0.1,
                        help = 'Learning rate decay at specified rounds')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--reg', default=1e-4, type=float, 
                        help='weight decay for an optimizer')
    parser.add_argument('--global_average',action = 'store_true',
                        help='use all clients (including which are not updated in this round) for averaging')
    parser.add_argument('--FedProx',action='store_true',
                        help='use FedProx')
    parser.add_argument('--FedBN',action='store_true',help="Whether to use FedBN")
    parser.add_argument('--mu',type = float, default=0.0,
                        help = 'mu in FedProx')
    parser.add_argument('--dynamic_mu',action = 'store_true',
                        help='use a dynamic mu for FedProx')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', 
                        help='model name')
    parser.add_argument('--pretrained',action="store_true",default=False,
                        help="Whether to use pretrained model from torchvision")
    
    # Adversarial Training
    parser.add_argument('--adv_warmup',type = int,default = 0,help='length of warm up phase')
    parser.add_argument('--adv_train', type = float, default = 0.0,
                        help = 'The fraction of clients that use adversarial samples for training')
    parser.add_argument('--adv_attack',type = str, choices=['PGD','BIM','FGSM','FGSM_RS'],default='PGD',
                        help = 'Kind of adversarial attack')
    parser.add_argument('--adv_epsilon',type = float,nargs="*", default=[0.15,])
    parser.add_argument('--adv_alpha',type = float,nargs="*", default=[0.01,])
    parser.add_argument('--adv_T', type = int, default=15)
    parser.add_argument('--adv_sample_ratio', type = float,default=1.0,
                        help = 'Ratio of adversarial training samples')
    parser.add_argument('--adv_cluster',action = 'store_true',help = "Clustered adversarial training")

    parser.add_argument('--adv_test', action = 'store_true',
                        help = 'Use adversarial samples for test')
    parser.add_argument('--advt_attack',type = str, choices=['PGD','BIM','FGSM','FGSM_RS'],default='PGD',
                        help = 'Kind of adversarial attack in test time')
    parser.add_argument('--advt_epsilon',type = float, default=0.15)
    parser.add_argument('--advt_alpha',type = float, default=0.01)
    parser.add_argument('--advt_T', type = int, default=15)
    
    # utils arguments
    parser.add_argument('--dataset', type=str, default='mnist', 
                        help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, 
                        help="number of classes")
    parser.add_argument('--gpu', default=None, 
                        help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', 
                        help="type of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--shards_per_client',type = int,default=1,
                        help='number of shards for each client')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unbalanced data splits for non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--alpha',type=float,default=None,
                        help="use Dirichlet_noniid sampling, set the alpha of Dir here")
    parser.add_argument('--verbose', type=int, default=1, 
                        help='verbose')
    parser.add_argument('--seed', type=int, default=None, nargs='*', 
                        help='random seed')
    parser.add_argument('--test_every', type=int, default=1,
                        help="test inference internal")
    parser.add_argument('--target_accuracy',type=float,default=None,
                        help='stop at a specified test accuracy')

    # system efficiency arguments
    parser.add_argument('--sys_efficiency_mode', type=str, default='bias', 
                        help='system efficiency mode: bias or energy-efficiency')
    parser.add_argument('--sys_gamma', type=float, default=1.0,
                        help='The hyperprameter gamma for system efficiency -> latency')
    parser.add_argument('--sys_theta', type=float, default=0,
                        help='The hyperparameter theta for system efficiency -> energy')
    parser.add_argument('--flsys_info', type=str, default='./flsys_profile_info',
                        help='The path of FL system information file')
    parser.add_argument('--random_device_seed', type=int, default=717,
                        help='random seed used for generate devices')    
    parser.add_argument('--rho',type=float,default=None,help="The hyperparameter rho for system efficiency")


    parser.add_argument('--strategy',type=str,default='rand',help="The selection strategy, default to rand")
    # GPR arguments
    parser.add_argument('--num_cluster', type = int, default=None, 
                        help = 'Number of clusters, if None, the cluster number will be dynamic')
    parser.add_argument('--clustering_th', type = float, default=None, 
                        help = 'The clustering distortion threshold, if None, the cluster number will be static')
    # parser.add_argument('--user_batch', type = int, default=1, 
    #                     help = 'batch size of user embeddings in GP training, only valid when cluster_gpr is set')
    # parser.add_argument('--cluster_batch', type = int, default=1, 
    #                     help = 'batch size of cluster embeddings in GP training, only valid when cluster_gpr is set')
    parser.add_argument('--gpr_gpu', default=None, 
                        help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--warmup',type = int, default=25,
                        help = 'length of warm up phase for GP')
    parser.add_argument('--gpr_begin',type = int,default=0,
                        help='the round begin to sample and train GP')
    # parser.add_argument('--group_size',type = int, default = 10, 
    #                     help = 'length of history round to sample for GP, equal to M in paper')
    parser.add_argument('--GPR_interval',type = int, default= 5, 
                        help = 'interval of sampling and training of GP, namely, Delta t')
    parser.add_argument('--GPR_gamma',type = float,default = 0.8,
                        help = 'gamma for training GP')
    # parser.add_argument('--GPR_Epoch',type=int,default=200,
    #                     help = 'number of optimization iterations of GP')
    parser.add_argument('--update_mean', action='store_true',help="Whether to update the mean of the GPR")
    parser.add_argument('--kernel',type = str,default = 'Poly',
                        help = 'kind of kernel used in GP (Poly,SE)')
    parser.add_argument('--poly_norm',type=int,default=0,
                        help='whether to normalize the poly kernel, set 1 to normalize')
    parser.add_argument('--dimension',type = int,default=15,
                        help = 'dimension of embedding in GP')
    parser.add_argument('--train_method',type = str,default='MML',
                        help = 'method of training GP (MML,LOO)')
    parser.add_argument('--noise',type = float,default=0.01,
                        help = 'noise added to the covariance matrix to avoid singularity')
    parser.add_argument('--discount',type = float, default=0.9, 
                        help = 'annealing coefficient, i.e., beta in paper')
    parser.add_argument('--greedy_epsilon',type = float,default=0.0,
                        help='use epsilon-greedy in FedGP, set epsilon here')
    parser.add_argument('--dynamic_C',action = 'store_true',
                        help = 'use dynamic GP clients selection')
    parser.add_argument('--dynamic_TH',type = float,default=0.0,
                        help='dynamic selection threshold')
    
    # Multivariate Normal Test arguments
    parser.add_argument('--mvnt',action='store_true',help='Perform Multivariate Normality Test')
    parser.add_argument('--mvn_samples', type = int, default=1000,help = 'Number of samples for each MVN Test')
    parser.add_argument('--mvn_dimensions',type = int, default= 100, help= 'Number of dimensions (clients) for MVN Test')
    parser.add_argument('--mvnt_interval',type = int, default=100, help='Interval for MVN Test')
    parser.add_argument('--mvnt_workers',type=int,default=8,help="Number of workers for MVN sampling")

    # Power-d arguments
    parser.add_argument('--d',type = int,default = 30,
                        help='d in Pow-d selection')

    # Active Federated Learning arguments
    parser.add_argument('--alpha1',type = float,default=0.75,
                        help = 'alpha_1 in ALF')
    parser.add_argument('--alpha2',type = float,default=0.01,
                        help = 'alpha_2 in AFL')
    parser.add_argument('--alpha3',type = float,default=0.1,
                        help='alpha_3 in AFL')

    # Oort arguments
    parser.add_argument('--pacer_step',type=float,default=0.001,help="Pacer step (Delta) of Oort")
    parser.add_argument('--step_window',type=int,default=20,help="Step window (W) of Oort")
    parser.add_argument('--epsilon_range',type=float,default=[0.9,0.2],nargs=2,help="The epsilon range of Oort, i.e., [init_eps, min_eps]")
    parser.add_argument('--epsilon_decay',type=float,default=0.98,help="The decay rate of epsilon in oort")
    parser.add_argument('--oort_alpha',type=float,default=2.0,help="The weight of systematic utility (alpha) in oort")
    parser.add_argument('--oort_c',type=float,default=0.95,help="The threshold (c) for accepting the utility in oort")

    # Harmony arguments
    parser.add_argument('--epsilon',type = float,default=0.5, help='epsilon in Harmony')
    parser.add_argument('--omega',type = float,default=1.0, help='omega in Harmony')
    parser.add_argument('--xi',type = float,default=math.sqrt(2), help='xi in Harmony')
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = args_parser()
    print(args.mlp_layers)
