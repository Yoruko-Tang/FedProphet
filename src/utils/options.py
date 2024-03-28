import argparse
import math

def args_parser():
    parser = argparse.ArgumentParser()

    fl_options(parser)
    
    fedcor_option(parser)
    powerd_option(parser)
    afl_option(parser)
    oort_option(parser)
    harmony_option(parser)
    
    at_option(parser)
    
    fedprophet_option(parser)
    kd_option(parser)
    
    args = parser.parse_args()
    return args


def fl_options(parser):
    ## basic federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--dataset', type=str, default='mnist', 
                        help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, 
                        help="number of classes")
    parser.add_argument('--gpu', default=None, 
                        help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--num_users', type=int, default=100,
                        help="total number of clients: N")
    

    # model arguments
    parser.add_argument('--model_arch', type=str, default='mlp', 
                        help='model name')
    parser.add_argument('--norm',type=str,choices=['BN','LN','IN','GN','None'],
                        default='BN',help='normalization type')
    parser.add_argument('--pretrained',action="store_true",default=False,
                        help="Whether to use pretrained model from torchvision")
    
    
    
    # statistical heterogeneity option
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--shards_per_client',type = int,default=1,
                        help='number of shards for each client')
    parser.add_argument('--skew', type=float, default=None,
                        help='the proportion of iid data in addition to SPC partition')
    parser.add_argument('--alpha',type=float,default=None,
                        help="use Dirichlet_noniid sampling, set the alpha of Dir here")
    
    # systematic heterogeneity option
    parser.add_argument('--flsys_profile_info', type=str, default='./src/hardware/flsys_profile_info',
                        help='The path of FL system information file')
    parser.add_argument('--device_random_seed', type=int, default=717,
                        help='random seed used for generate devices')
    parser.add_argument('--sys_scaling_factor',type=float,default=0.0,
                        help='the factor that controls the distribution of different devices')
    parser.add_argument('--reserved_flops',type=float,default=None,
                        help='the maximum number of flops allowed in model partition')
    parser.add_argument('--reserved_mem',type=float,default=None,
                        help='the maximum memory allowed in model partition')
    
    ## FL algorithm options
    parser.add_argument('--flalg',type=str,default='FedAvg',
                        help="The algorithm for FL optimizer")
    

    # selector arguments
    parser.add_argument('--strategy',type=str,default='rand',
                        choices=['rand','afl','powerd','oort','fedcbs','harmony','fedcor','cfedcor'],
                        help="The selection strategy, default to rand")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    
    # optimizer args
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--target_accuracy',type=float,default=None,
                        help='stop at a specified test accuracy')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay',type = float,default=None,
                        help = 'Learning rate decay at specified rounds')
    parser.add_argument('--lr_schedule', type=int, nargs='*', default=None,
                        help='Decrease learning rate at these rounds.')
    parser.add_argument('--optimizer', type=str, default='sgd', 
                        choices=['sgd','adam'],
                        help="type of optimizer")
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--reg', default=1e-4, type=float, 
                        help='weight decay for an optimizer')

    

    ## util args
    parser.add_argument('--test_every', type=int, default=1,
                        help="test inference internal")
    parser.add_argument('--verbose', action='store_true', default=False, 
                        help='whether to print the training procudure of each client')
    parser.add_argument('--seed', type=int, default=None, nargs='*', 
                        help='random seed')

def fedcor_option(parser):
    
    # FedCor and FedRepre arguments
    parser.add_argument('--num_cluster', type = int, default=None, 
                        help = 'Number of clusters, if None, the cluster number will be dynamic')
    parser.add_argument('--clustering_th', type = float, default=None, 
                        help = 'The clustering distortion threshold, if None, the cluster number will be static')
    parser.add_argument('--gpr_gpu', default=None, 
                        help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--warmup',type = int, default=25,
                        help = 'length of warm up phase for GP')
    parser.add_argument('--gpr_begin',type = int,default=0,
                        help='the round begin to sample and train GP')
    parser.add_argument('--GPR_interval',type = int, default= 5, 
                        help = 'interval of sampling and training of GP, namely, Delta t')
    parser.add_argument('--GPR_gamma',type = float,default = 0.8,
                        help = 'gamma for training GP')
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

def powerd_option(parser):
    # Power-d arguments
    parser.add_argument('--d',type = int,default = 30,
                        help='d in Pow-d selection')

def afl_option(parser):
    # Active Federated Learning arguments
    parser.add_argument('--alpha1',type = float,default=0.75,
                        help = 'alpha_1 in ALF')
    parser.add_argument('--alpha2',type = float,default=0.01,
                        help = 'alpha_2 in AFL')
    parser.add_argument('--alpha3',type = float,default=0.1,
                        help='alpha_3 in AFL')

def oort_option(parser):
    # Oort arguments
    parser.add_argument('--pacer_step',type=float,default=0.001,help="Pacer step (Delta) of Oort")
    parser.add_argument('--step_window',type=int,default=20,help="Step window (W) of Oort")
    parser.add_argument('--epsilon_range',type=float,default=[0.9,0.2],nargs=2,help="The epsilon range of Oort, i.e., [init_eps, min_eps]")
    parser.add_argument('--epsilon_decay',type=float,default=0.98,help="The decay rate of epsilon in oort")
    parser.add_argument('--oort_alpha',type=float,default=2.0,help="The weight of systematic utility (alpha) in oort")
    parser.add_argument('--oort_c',type=float,default=0.95,help="The threshold (c) for accepting the utility in oort")

def harmony_option(parser):
    # Harmony arguments
    parser.add_argument('--epsilon',type = float,default=0.5, help='epsilon in Harmony')
    parser.add_argument('--omega',type = float,default=1.0, help='omega in Harmony')
    parser.add_argument('--xi',type = float,default=math.sqrt(2), help='xi in Harmony')


def at_option(parser):
    # AT args
    # Adversarial Training
    parser.add_argument('--adv_train', action = 'store_true',
                        help = 'Use adversarial samples for training')
    parser.add_argument('--adv_method',type = str, choices=['PGD','BIM','FGSM','FGSM_RS'],default='PGD',
                        help = 'Kind of adversarial attack')
    parser.add_argument('--adv_epsilon',type = float, default=8/255)
    parser.add_argument('--adv_alpha',type = float, default=2/255)
    parser.add_argument('--adv_T', type = int, default=10)
    parser.add_argument('--adv_norm', type = str, default='inf')
    parser.add_argument('--adv_bound',type = float,nargs=2, default=[0.0,1.0])
    parser.add_argument('--adv_warmup',type = int,default = 0,help='length of warm up phase')
    parser.add_argument('--warmup_adv_ratio', type = float,default=0.0,
                        help = 'Ratio of adversarial training samples in warmup phase')
    parser.add_argument('--adv_ratio', type = float,default=1.0,
                        help = 'Ratio of adversarial training samples after warmup phase')
    
    # Adversarial Test
    parser.add_argument('--adv_test', action = 'store_true',
                        help = 'Use adversarial samples for test')
    parser.add_argument('--advt_method',type = str, choices=['PGD','BIM','FGSM','FGSM_RS'],default='PGD',
                        help = 'Kind of adversarial attack in test time')
    parser.add_argument('--advt_epsilon',type = float, default=8/255)
    parser.add_argument('--advt_alpha',type = float, default=2/255)
    parser.add_argument('--advt_T', type = int, default=10)
    parser.add_argument('--advt_norm', type = str, default='inf')
    parser.add_argument('--advt_bound',type = float,nargs=2, default=[0.0,1.0])


def fedprophet_option(parser):
    # FedProphet Args
    parser.add_argument('--mu',type = float,default=0.0, help='mu in fedprophet')
    parser.add_argument('--lamb',type = float,default=0.0, help='lambda in fedprophet')
    parser.add_argument('--psi',type = float,default=0.0, help='psi in fedprophet')
    parser.add_argument('--eps_quantile',type = float,default=0.1, help='quantile for choosing the epsilon')
    parser.add_argument('--adapt_eps',action = 'store_true',default=False,help = "adaptively adjust the eps_quantile during training")
    parser.add_argument('--int_adv_norm', type = str, choices=['inf','l2'],default='l2')
    parser.add_argument('--stage_lr_decay',type=float,default=None,help="decay learning rate during stage forward")


def kd_option(parser):
    # FedET and FedDF Args
    parser.add_argument('--public_dataset_size',type=int,default=5000,help="number of data in the public set")
    parser.add_argument('--dist_iters',type=int,default=128,help="number of distillation iterations")
    parser.add_argument('--dist_lr',type=float,default=5e-3,help="learning rate for distillation")
    parser.add_argument('--dist_batch_size',type=int,default=64,help="batch size for distillation")
    parser.add_argument('--diver_lamb',type=float,default=0.05,help="weight of diversity loss")

