import datasets

from datasets.sampling import iid, SPC_noniid,SPC_skew_noniid,Dirichlet_noniid

from torch.utils.data import Subset
import numpy as np
from numpy.random import RandomState


def get_dataset(args,seed=None):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    rs = RandomState(seed)
    assert args.dataset in datasets.__dict__.keys(), "The dataset {} is not supported!".format(args.dataset)
    
    if datasets.dataset_to_datafamily[args.dataset] in ['mnist','cifar','imagenet']: # CV datasets
        dataset = datasets.__dict__[args.dataset]
        modelfamily = datasets.dataset_to_datafamily[args.dataset]
        train_transform = datasets.datafamily_to_transforms[modelfamily]['adv_train' if args.adv_train else 'train']
        test_transform = datasets.datafamily_to_transforms[modelfamily]['adv_test' if args.adv_test else 'test']
        train_dataset = dataset(train=True, transform=train_transform,download=True)
        test_dataset = dataset(train=False, transform=test_transform,download=True)
        args.num_classes = len(train_dataset.classes)
    
        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = iid(train_dataset, args.num_users,rs)
            #user_groups_test = iid(test_dataset,args.num_users,rs)
        else:
            # Sample Non-IID user data from Mnist
            if args.alpha is not None:
                user_groups = Dirichlet_noniid(train_dataset, args.num_users,args.alpha,rs,minimal_datasize=len(train_dataset)//(args.num_users*10))
                #user_groups_test = Dirichlet_noniid(test_dataset, args.num_users,args.alpha,rs,minimal_datasize=len(test_dataset)//(args.num_users*10))
            else:
                # Chose euqal splits for every user
                if args.skew is None:
                    user_groups = SPC_noniid(train_dataset, args.num_users,args.shards_per_client,rs)
                    #user_groups_test = SPC_noniid(test_dataset, args.num_users,args.shards_per_client,rs)
                else:
                    user_groups = SPC_skew_noniid(train_dataset, args.num_users,args.shards_per_client,args.skew,rs)
                    #user_groups_test = SPC_skew_noniid(test_dataset, args.num_users,args.shards_per_client,args.skew,rs)


    elif datasets.dataset_to_datafamily[args.dataset] == 'nlp':
        if args.dataset == 'shakespeare':
            args.num_classes = 80
            
        elif args.dataset == 'sent140':
            args.num_classes = 2
        else:
            raise RuntimeError("Not registered NLP dataset!")  
        data_dir = './data/{}/'.format(args.dataset)
        #user_groups_test={}
        train_dataset,test_dataset,user_groups=datasets.__dict__[args.dataset](data_dir,args.shards_per_client,rs)
        
    else:
        raise RuntimeError("Not registered dataset!")
    
    args.num_users=len(user_groups.keys())
    weights = []
    sizes = []
    for i in range(args.num_users):
        weights.append(len(user_groups[i])/len(train_dataset))
        sizes.append(len(user_groups[i]))
    
    print("====> Complete dataset partitioning!")
    print("====> Minimum local datasize: {}, maximum local datasize: {}".format(np.min(sizes),
                                                                                np.max(sizes)))
    print("====> Total training datasize: {}, Total test datasize: {}".format(np.sum(sizes),
                                                                              len(test_dataset)))
    
    return train_dataset, test_dataset, user_groups, np.array(weights)

def get_data_matrix(dataset,user_groups,num_classes):
    num_users = len(user_groups.keys())
    data_matrix = np.zeros([num_users,num_classes],dtype = np.int64)
    for i in range(num_users):
        subset = Subset(dataset,user_groups[i])
        for _,label in subset:
            data_matrix[i,label] = data_matrix[i,label] + 1
    return data_matrix