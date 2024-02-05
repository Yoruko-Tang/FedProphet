#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from cvxopt import matrix,solvers


def iid(dataset, num_users,rs):
    """
    Sample I.I.D. client data
    """
    num_items = int(len(dataset)//num_users)
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(rs.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def SPC_noniid(dataset, num_users,shards_per_client,rs):
    """
    Sample non-I.I.D client data with equal number of shards for each client and equal size of  each shard.
    """
    num_shards = shards_per_client*num_users
    num_imgs =  len(dataset)//num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([],dtype=np.int64) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(rs.choice(idx_shard, shards_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        rs.shuffle(dict_users[i])
    return dict_users

def SPC_skew_noniid(dataset,num_users,shards_per_client,skew,rs):
    """
    sampling non-iid data distribution with (1-skew) data from shards_noniid and (skew) data from iid
    """
   
    non_iid_idx = rs.choice(np.arange(len(dataset)),int((1-skew)*len(dataset)),replace=False)
    iid_idx = list(set(range(len(dataset)))-set(non_iid_idx))
    num_iid_imgs = len(iid_idx)//num_users

    # build non-iid data distribution
    num_shards = shards_per_client*num_users
    num_non_iid_imgs = len(non_iid_idx)//num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([],dtype=np.int64) for i in range(num_users)}
    # idxs = np.arange(num_shards*num_imgs)
    try:
        labels = np.array(dataset.targets)[non_iid_idx]
    except:
        labels = np.array(dataset.labels)[non_iid_idx]


    # sort labels
    idxs_labels = np.vstack((non_iid_idx, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(rs.choice(idx_shard, shards_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_non_iid_imgs:(rand+1)*num_non_iid_imgs]), axis=0)
        # append iid part
        iid_set = rs.choice(iid_idx, num_iid_imgs, replace=False)
        iid_idx = list(set(iid_idx)-set(iid_set))
        dict_users[i] = np.concatenate((dict_users[i], iid_set), axis=0)
        rs.shuffle(dict_users[i])
        
    
    return dict_users

def Dirichlet_noniid(dataset,num_users,alpha,rs,minimal_datasize=1):
    """
    Sample dataset with dirichlet distribution and concentration parameter alpha
    """
    # img_num_per_client = len(dataset)//num_users
    dict_users = {i: np.array([],dtype=np.int64) for i in range(num_users)}
    idxs = np.arange(len(dataset))
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    num_classes = len(dataset.classes)
    labels_idxs = []
    prior_class_distribution = np.zeros(num_classes)
    d = np.zeros(num_classes)
    for i in range(num_classes):
        labels_idxs.append(idxs[labels==i])
        prior_class_distribution[i] = len(labels_idxs[i])/len(dataset)
        d[i]=len(labels_idxs[i])
    
    data_ratio = np.zeros([num_classes,num_users])
    if isinstance(alpha,list):
        for i in range(num_users):
            data_ratio[:,i] = rs.dirichlet(prior_class_distribution*alpha[i])
    else:
        data_ratio = np.transpose(rs.dirichlet(prior_class_distribution*alpha,size=num_users))
    # data_ratio = data_ratio/np.sum(data_ratio,axis=1,keepdims=True)
    # Client_DataSize = len(dataset)//num_users*np.ones([num_users,1],dtype=np.int64)
    A = matrix(data_ratio)
    b = matrix(d)
    G = matrix(-np.eye(num_users))
    h = matrix(-minimal_datasize*np.ones([num_users,1]))
    P = matrix(np.eye(num_users))
    q = matrix(np.zeros([num_users,1]))
    results = solvers.qp(P,q,G,h,A,b)
    if results['status'] == 'unknown':# the original optimization is not feasible
        # we relax the equation constraint and try to find the solution that 
        # is the closest to the feasible area
        P = matrix(np.transpose(data_ratio)@data_ratio)
        q = matrix(-np.transpose(data_ratio)@np.reshape(d,[-1,1]))
        G = matrix(np.vstack([data_ratio,-np.eye(num_users)]))
        h = matrix(np.vstack([np.reshape(d,[-1,1]),-minimal_datasize*np.ones([num_users,1])]))
        results = solvers.qp(P,q,G,h)
    Client_DataSize = np.array(results['x'])
    Data_Division = data_ratio*np.transpose(Client_DataSize)
    rest = []
    for label in range(num_classes):
        for client in range(num_users):
            data_idx = rs.choice(labels_idxs[label],int(Data_Division[label,client]),replace=False)
            dict_users[client] = np.concatenate([dict_users[client],data_idx],0)
            labels_idxs[label] = list(set(labels_idxs[label])-set(data_idx))
        rest = rest+labels_idxs[label]

    rest_clients = rs.choice(range(num_users),len(rest),replace = True)
    
    for n,user in enumerate(rest_clients):
        dict_users[user] = np.append(dict_users[user],rest[n])

    for user in range(num_users):
        rs.shuffle(dict_users[user])

    return dict_users


    


