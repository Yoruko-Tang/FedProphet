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

    if not args.iid:
        file_name = f"./save/{args.dataset}_{args.model_arch}_{args.epochs}_N[{args.num_users}]_{'all' if args.frac == 1.0 else args.strategy}[{args.frac}]_{'sp' if args.alpha is None else 'alpha'}[{args.shards_per_client+(args.skew if args.skew is not None else 0) if args.alpha is None else args.alpha}]_sys[{args.sys_scaling_factor}]_E[{args.local_ep}]_B[{args.local_bs}]_lr[{args.lr}({args.lr_decay})]"
    else:
        file_name = f"./save/{args.dataset}_{args.model_arch}_{args.epochs}_N[{args.num_users}]_{'all' if args.frac == 1.0 else args.strategy}[{args.frac}]_iid_sys[{args.sys_scaling_factor}]_E[{args.local_ep}]_B[{args.local_bs}]_lr[{args.lr}({args.lr_decay})]"
    
    file_name = os.path.join(file_name,args.flalg)
    if int(args.adv_train)*args.adv_ratio > 0:
        file_name += "AT{}[eps{}_T{}_tar{}]".format(int(args.adv_train)*args.adv_ratio,args.adv_epsilon,args.adv_T,args.target_clean_adv_ratio)

    if args.flalg == 'FedProphet':
        file_name = os.path.join(file_name,"mu{}_lambda{}_psi{}_quan{}_ada{}_int01".format(args.mu,args.lamb,args.psi,args.eps_quantile,int(args.adapt_eps)))
    
    
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    

    return file_name


def exp_details(args):
    print('\nExperimental details:')
    print('    Model     : {}'.format(args.model_arch))
    print('    Optimizer : {}'.format(args.optimizer))
    print('    Learning  : {}'.format(args.lr))
    print('    Global Rounds   : {}'.format(args.epochs))

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print(f"    Non-IID: {'sp' if args.alpha is None else 'alpha'}={args.shards_per_client+(args.skew if args.skew is not None else 0) if args.alpha is None else args.alpha}")
    print('    Fraction of users  : {}'.format(args.frac))
    print('    Local Batch size   : {}'.format(args.local_bs))
    print('    Local Epochs       : {}\n'.format(args.local_ep))
    
    print('    Algorithm    : {}'.format(args.flalg))
    print('    Selection Strategy    : {}'.format(args.strategy))
    return


