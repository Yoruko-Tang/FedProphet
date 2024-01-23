#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import os.path as osp
import os
import torchvision.models
import types

def get_net(modelname, modeltype, pretrained=False, num_classes=1000):
    """
    modelname: must be exactly the same as the classes in torchvision.models
    e.g., vgg11, vgg16
    """
    assert "vgg" in modelname, "Only support VGG for pretrained model currently"
    assert modeltype in ["imagenet","cifar"], "Only support imagenet-like or cifar-like datasets with pretrained model currently"
    model = eval('torchvision.models.{}'.format(modelname))(pretrained=pretrained)
    if num_classes!=1000: # reinitialize the last layer
        if modeltype == 'imagenet': # for 224x224x3 inputs
            in_feat = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_feat, num_classes)
        else: # for 32x32x3 inputs
            if hasattr(model,"avgpool"):
                model.avgpool = nn.AdaptiveAvgPool2d((1,1))
            in_feat = model.classifier[0].in_features//49
            model.classifier = nn.Sequential(
                nn.Linear(in_feat, in_feat),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(in_feat, in_feat),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(in_feat, num_classes))# use less neurons for a small input
            #model.classifier = nn.Linear(in_feat, num_classes)

    return modify_pretrained_model(model)
    

def modify_pretrained_model(model):
    def Get_Local_State_Dict(self):
        sd = self.state_dict()
        for name in list(sd.keys()):
            if 'weight' in name or 'bias' in name:
                sd.pop(name)
        return sd

    def Load_Local_State_Dict(self,local_dict):
        # load local parameters saved by Get_Local_State_Dict()
        sd = self.state_dict()
        sd.update(local_dict)
        self.load_state_dict(sd)

    model.Get_Local_State_Dict = types.MethodType(Get_Local_State_Dict,model)
    model.Load_Local_State_Dict = types.MethodType(Load_Local_State_Dict,model)
    return model


class FedModule(object):
    def __init__(self) -> None:
        pass

    def Get_Local_State_Dict(self):
        sd = self.state_dict()
        for name in list(sd.keys()):
            if 'weight' in name or 'bias' in name:
                sd.pop(name)
        return sd

    def Load_Local_State_Dict(self,local_dict):
        # load local parameters saved by Get_Local_State_Dict()
        sd = self.state_dict()
        sd.update(local_dict)
        self.load_state_dict(sd)


class LeNet5(nn.Module,FedModule):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)
    



class MLP(nn.Module,FedModule):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.dim_in = dim_in
        self.layers = []
        self.relus = []
        self.dropouts = []
        
        if len(dim_hidden)>0:
            self.layers.append(nn.Linear(dim_in, dim_hidden[0]))
            self.relus.append(nn.ReLU())
            self.dropouts.append(nn.Dropout())
            for n in range(len(dim_hidden)-1):
                self.layers.append(nn.Linear(dim_hidden[n],dim_hidden[n+1]))
                self.relus.append(nn.ReLU())
                self.dropouts.append(nn.Dropout())
            self.layers.append(nn.Linear(dim_hidden[-1], dim_out))
        else:
            # logistic regression
            self.layers.append(nn.Linear(dim_in, dim_out))
        

        self.layers = nn.ModuleList(self.layers)
        self.relus = nn.ModuleList(self.relus)
        self.dropouts = nn.ModuleList(self.dropouts)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        for n in range(len(self.relus)):
            x = self.layers[n](x)
            x = self.dropouts[n](x)
            x = self.relus[n](x)
        x = self.layers[-1](x)
        return x



                    

class NaiveCNN(nn.Module,FedModule):
    def __init__(self, args,input_shape = [3,32,32],final_pool=True):
        super(NaiveCNN, self).__init__()
        self.convs = []
        self.fcs = []
        self.final_pool=final_pool
        if len(args.kernel_sizes) < len(args.num_filters):
            exlist = [args.kernel_sizes[-1] for i in range(len(args.num_filters)-len(args.kernel_sizes))]
            args.kernel_sizes.extend(exlist)
        elif len(args.kernel_sizes) > len(args.num_filters):
            exlist = [args.num_filters[-1] for i in range(len(args.kernel_sizes)-len(args.num_filters))]
            args.num_filters.extend(exlist)
        output_shape = np.array(input_shape)
        for ksize in args.kernel_sizes[:-1] if not final_pool else args.kernel_sizes:
            if args.padding:
                pad = ksize//2
                output_shape[1:] = (output_shape[1:]+2*pad-ksize-1)//2+1
            else:
                output_shape[1:] = (output_shape[1:]-ksize-1)//2+1
        if not final_pool:
            if args.padding:
                pad = args.kernel_sizes[-1]//2
                output_shape[1:] = output_shape[1:]+2*pad-args.kernel_sizes[-1]+1
            else:
                output_shape[1:] = output_shape[1:]-args.kernel_sizes[-1]+1
        output_shape[0] = args.num_filters[-1]
        conv_out_length = output_shape[0]*output_shape[1]*output_shape[2]
        
        self.convs.append(nn.Conv2d(input_shape[0], args.num_filters[0], kernel_size=args.kernel_sizes[0],padding = args.kernel_sizes[0]//2 if args.padding else 0))
        for n in range(len(args.num_filters)-1):
            self.convs.append(nn.Conv2d(args.num_filters[n], args.num_filters[n+1], kernel_size=args.kernel_sizes[n+1],padding = args.kernel_sizes[n+1]//2 if args.padding else 0))
        #self.conv2_drop = nn.Dropout2d()
        self.fcs.append(nn.Linear(conv_out_length, args.mlp_layers[0]))
        for n in range(len(args.mlp_layers)-1):
            self.fcs.append(nn.Linear(args.mlp_layers[n], args.mlp_layers[n+1]))
        self.fcs.append(nn.Linear(args.mlp_layers[-1], args.num_classes))
        
        self.convs = nn.ModuleList(self.convs)
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, x):
        for n in range(len(self.convs)-1 if not self.final_pool else len(self.convs)):
            x = F.relu(F.max_pool2d(self.convs[n](x), 2))
        if not self.final_pool:
            x = F.relu(self.convs[-1](x))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        for n in range(len(self.fcs)-1):
            x = F.relu(self.fcs[n](x))
        x = self.fcs[-1](x)
        return x
        

    
class BNCNN(NaiveCNN):
    def __init__(self, args,input_shape = [1,28,28]):
        super(BNCNN, self).__init__(args,input_shape)
        self.bns = []
        for num_filter in args.num_filters:
            self.bns.append(nn.BatchNorm2d(num_filter))
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        for n in range(len(self.convs)):
            x = F.relu(F.max_pool2d(self.bns[n](self.convs[n](x)), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        for n in range(len(self.fcs)-1):
            x = F.relu(self.fcs[n](x))
            #x = F.dropout(x, training=self.training)
        x = self.fcs[-1](x)
        return x


class RNN(nn.Module):
    def __init__(self, n_hidden, n_classes,embedding=8,pretrain_emb=False,fc_layer = None):
        super(RNN, self).__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.fc_layer = fc_layer
        if not pretrain_emb:
            self.emb = nn.Embedding(n_classes, embedding)
        else:
            self.emb = None
        self.LSTM = nn.LSTM(embedding, n_hidden, 2)
        if fc_layer is not None:
            self.hidden_fc = nn.Linear(n_hidden,fc_layer)
            self.fc = nn.Linear(fc_layer, n_classes)
        else:
            self.fc = nn.Linear(n_hidden, n_classes)

    def forward(self, features):
        if self.emb is not None: # [batch,seq_len]
            x = self.emb(features.T) #[seq_len, batch, emb_len]
        else: # [batch, seq_len, emb_len]
            x = torch.transpose(features,0,1) #[seq_len, batch, emb_len]
        x, _ = self.LSTM(x) #[seq_len, batch, n_hidden]
        if self.fc_layer is not None:
            x = self.hidden_fc(x[-1, :, :])#[batch, fc_layer]
            x = self.fc(x) #[batch, n_classes]
        else:
            x = self.fc(x[-1, :, :]) #[batch, n_classes]
        return x



if __name__ == '__main__':
    rnn = RNN(8, 256, 80)
    sd = rnn.state_dict()
    for name in list(sd.keys()):
        print(name)
    