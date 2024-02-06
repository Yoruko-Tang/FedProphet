#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# import os
# import sys
# sys.path.append(os.path.join(os.getcwd(),"src"))

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

import models
from datasets import datafamily_to_normalize




def get_net(modelname, modeltype, num_classes=1000, 
            pretrained=False, adv_norm=False, modularization=False):
    """
    modelname: must be exactly the same as the classes in torchvision.models
    e.g., vgg11, vgg16
    """
    # assert "vgg" in modelname or 'resnet' in modelname, "Only support VGG and ResNet for pretrained model currently"
    if pretrained:
        assert modeltype in ["imagenet","cifar"], "Only support imagenet-like or cifar-like datasets with pretrained model currently"
    
    # get pretrained model
    model = eval('models.{}'.format(modelname))(weights="DEFAULT" if pretrained else None)
    # adapt to the specified num_classes
    model = eval('models.{}.adapt'.format(models.modelname_to_modelfamily(modelname)))(model,modeltype,num_classes)
    # add list of feature layers for memory tracking
    model = eval('models.{}.set_feature_layer'.format(models.modelname_to_modelfamily(modelname)))(model)


    # add normalization layer to the adversarial training model
    if adv_norm:
        norm_info = datafamily_to_normalize[modeltype]
        mean,std = norm_info["mean"],norm_info["std"]
        normalization_layer = Normalize(mean,std)
        model = eval('models.{}.add_normalization'.format(models.modelname_to_modelfamily(modelname)))(model,normalization_layer)
    
    # modularize the model such that the model can enter and exit at any layers
    if modularization:
        model = eval('models.{}.modularization'.format(models.modelname_to_modelfamily(modelname)))(model)
    return model
    


class Normalize(nn.Module):
    def __init__(self,mean,std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean).reshape([1,-1,1,1])
        self.std = torch.tensor(std).reshape([1,-1,1,1])
    
    def forward(self,x):
        self.mean = self.mean.to(x)
        self.std = self.std.to(x)
        return (x-self.mean)/self.std

class DualBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.register_buffer('adv_running_mean',self.running_mean.clone())
        self.register_buffer('adv_running_var',self.running_var.clone())
        self.register_buffer('adv_num_batches_tracked',self.num_batches_tracked.clone())

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
            try:
                self.adv_running_mean.zero_()
                self.adv_running_var.fill_(1)
                self.num_batches_tracked.zero_()
            except:
                pass

    def clone_clean_stat(self):
        self.adv_running_mean = self.running_mean.clone()
        self.adv_running_var = self.running_var.clone()

    def forward(self,input:torch.Tensor,adv = False):
        """
        From Source code of Pytorch 1.7.1
        """
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                if not adv:
                    self.num_batches_tracked = self.num_batches_tracked + 1
                else:
                    if self.adv_num_batches_tracked == 0: # reinitialize the adv stat with the current clean stat
                        self.clone_clean_stat()
                    self.adv_num_batches_tracked = self.adv_num_batches_tracked + 1
                    
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked if not adv else self.adv_num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            if not adv:
                bn_training = (self.running_mean is None) and (self.running_var is None)
            else:
                bn_training = (self.adv_running_mean is None) and (self.adv_running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if not adv:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
        else:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.adv_running_mean if not self.training or self.track_running_stats else None,
                self.adv_running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)


class DualBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.register_buffer('adv_running_mean',self.running_mean.clone())
        self.register_buffer('adv_running_var',self.running_var.clone())
        self.register_buffer('adv_num_batches_tracked',self.num_batches_tracked.clone())
    
    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
            try:
                self.adv_running_mean.zero_()
                self.adv_running_var.fill_(1)
                self.num_batches_tracked.zero_()
            except:
                pass
    
    def clone_clean_stat(self):
        self.adv_running_mean = self.running_mean.clone()
        self.adv_running_var = self.running_var.clone()

    def forward(self,input:torch.Tensor,adv = False):
        """
        From Source code of Pytorch 1.7.1
        """
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                if not adv:
                    self.num_batches_tracked = self.num_batches_tracked + 1
                else:
                    if self.adv_num_batches_tracked == 0: # reinitialize the adv stat with the current clean stat
                        self.clone_clean_stat()
                    self.adv_num_batches_tracked = self.adv_num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked if not adv else self.adv_num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            if not adv:
                bn_training = (self.running_mean is None) and (self.running_var is None)
            else:
                bn_training = (self.adv_running_mean is None) and (self.adv_running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if not adv:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
        else:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.adv_running_mean if not self.training or self.track_running_stats else None,
                self.adv_running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)



class FedModule(nn.Module):
    def __init__(self,normalization=None,input_normalize_mean = None,input_normalize_var = None):
        super().__init__()
        self.fcs = []
        self.convs = []
        self.max_pools = []
        self.normals = []
        self.normalization = normalization
        if input_normalize_mean is not None and input_normalize_var is not None:
            self.input_normalize = Normalize(input_normalize_mean,input_normalize_var)
        else:
            self.input_normalize = None    

    def forward(self, x,enter_stage = 0,early_exit=None,early_stop=False,adv=False):
        if self.input_normalize is not None:
            x = self.input_normalize(x)
        if enter_stage is None:
            enter_stage = 0
        int_exit = isinstance(early_exit,int)
        if int_exit:
            early_exit=[early_exit,]
        stage = 0
        early_exit_output=[]
        if x.dim()==4:
            if enter_stage < len(self.convs):
                for n in range(len(self.convs)):
                    if stage<enter_stage:
                        stage += 1
                        continue
                    x = self.convs[n](x)
                    if self.normalization is not None and self.normalization is not None:
                        if 'Dual' not in self.normalization:
                            x = self.normals[n](x)
                        else:
                            x = self.normals[n](x,adv)
                    x = F.relu(x)
                    if self.max_pools[n] is not None:
                        x = self.max_pools[n](x)

                    stage += 1
                    if early_exit is not None and stage in early_exit:
                        early_exit_output.append(x.clone())
                        if early_stop:
                            return early_exit_output[-1]
            else:
                stage = len(self.convs)
            x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        for n in range(len(self.fcs)-1):
            if stage<enter_stage:
                stage += 1
                continue
            if self.normalization is not None and "BatchNorm" in self.normalization:
                x = self.fcs[n](x)
                if 'Dual' not in self.normalization:
                    x = self.normals[n+len(self.convs)](x)
                else:
                    x = self.normals[n+len(self.convs)](x,adv)
                x = F.relu(x)
            else:
                x = F.dropout(x, training=self.training)
                x = F.relu(self.fcs[n](x))
            stage += 1
            if early_exit is not None and stage in early_exit:
                early_exit_output.append(x.clone())
                if early_stop:
                    return early_exit_output[-1]
        x = self.fcs[-1](x)
        if early_exit is not None:
            return x,early_exit_output if not int_exit else early_exit_output[0]
        return x


    def Get_Local_State_Dict(self):
        # save local parameters without weights and bias
        sd = self.state_dict()
        for name in list(sd.keys()):
            if 'normals' not in name:
                sd.pop(name)
        return sd

    def Load_Local_State_Dict(self,local_dict):
        # load local parameters saved by Get_Local_State_Dict()
        sd = self.state_dict()
        sd.update(local_dict)
        self.load_state_dict(sd)

    

    def get_layer_state_dict(self,layers,normalization_params=False):
        # Return the state dict with None on the other layers
        layer_state_dict = {k:None for k in self.state_dict().keys()}
        for l in layers:
            if hasattr(self,'convs') and l < len(self.convs):
                layer_state_dict['convs.%d.weight'%l]=self.state_dict()['convs.%d.weight'%l]
                layer_state_dict['convs.%d.bias'%l]=self.state_dict()['convs.%d.bias'%l]
                
            elif hasattr(self,'convs') and l < len(self.fcs)+len(self.convs):
                layer_state_dict['fcs.%d.weight'%(l-len(self.convs))]=self.state_dict()['fcs.%d.weight'%(l-len(self.convs))]
                layer_state_dict['fcs.%d.bias'%(l-len(self.convs))]=self.state_dict()['fcs.%d.bias'%(l-len(self.convs))]
            elif l < len(self.fcs):
                layer_state_dict['fcs.%d.weight'%(l)]=self.state_dict()['fcs.%d.weight'%(l)]
                layer_state_dict['fcs.%d.bias'%(l)]=self.state_dict()['fcs.%d.bias'%(l)]
            if self.normalization is not None and normalization_params:
                if l < len(self.normals):
                    layer_state_dict['normals.%d.weight'%l]=self.state_dict()['normals.%d.weight'%l]
                    layer_state_dict['normals.%d.bias'%l]=self.state_dict()['normals.%d.bias'%l]
                    if 'BatchNorm' in self.normalization: # for Batch Norm and Dual Batch Norm
                        layer_state_dict['normals.%d.running_mean'%l]=self.state_dict()['normals.%d.running_mean'%l]
                        layer_state_dict['normals.%d.running_var'%l]=self.state_dict()['normals.%d.running_var'%l]
                        layer_state_dict['normals.%d.num_batches_tracked'%l]=self.state_dict()['normals.%d.num_batches_tracked'%l]
                        if self.normalization == 'DualBatchNorm':
                            layer_state_dict['normals.%d.adv_running_mean'%l]=self.state_dict()['normals.%d.adv_running_mean'%l]
                            layer_state_dict['normals.%d.adv_running_var'%l]=self.state_dict()['normals.%d.adv_running_var'%l]
                            layer_state_dict['normals.%d.adv_num_batches_tracked'%l]=self.state_dict()['normals.%d.adv_num_batches_tracked'%l]
        return layer_state_dict

    def get_early_exit_shape(self,exit_layer):
        test = torch.zeros(self.input_shape).unsqueeze(0).to(self.fcs[0].weight)
        output = self.forward(test,early_exit=exit_layer,early_stop=True)
        return output.shape



class MLP(FedModule):
    def __init__(self, dim_in, dim_hidden, dim_out,normalization=None,input_normalize_mean = None,input_normalize_var = None):
        super().__init__(normalization,input_normalize_mean,input_normalize_var)



        if len(dim_hidden)>0:
            self.fcs.append(nn.Linear(dim_in, dim_hidden[0]))
            if self.normalization == "BatchNorm":
                self.normals.append(nn.BatchNorm1d(dim_hidden[0]))
            elif self.normalization == "DualBatchNorm":
                self.normals.append(DualBatchNorm1d(dim_hidden[0]))
            for n in range(len(dim_hidden)-1):
                self.fcs.append(nn.Linear(dim_hidden[n],dim_hidden[n+1]))
                if self.normalization == "BatchNorm":
                    self.normals.append(nn.BatchNorm1d(dim_hidden[n+1]))
                elif self.normalization == "DualBatchNorm":
                    self.normals.append(DualBatchNorm1d(dim_hidden[n+1]))
                
            self.fcs.append(nn.Linear(dim_hidden[-1], dim_out)) 

        else:
            # logistic regression
            self.fcs.append(nn.Linear(dim_in, dim_out))
        
        self.fcs = nn.ModuleList(self.fcs)
        if self.normalization is not None:
            self.normals = nn.ModuleList(self.normals)

        self.depth = len(self.fcs)
        
               

class NaiveCNN(FedModule):
    def __init__(self, args,input_shape = [3,32,32],final_pool=True,normalization=None,input_normalize_mean = None,input_normalize_var = None):
        super().__init__(normalization,input_normalize_mean,input_normalize_var)
        
        self.input_shape = input_shape

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
        self.max_pools.append(nn.MaxPool2d(2))
        if self.normalization == 'GroupNorm':
            self.normals.append(nn.GroupNorm(math.ceil(args.num_filters[0]/4),args.num_filters[0]))
        elif self.normalization == 'BatchNorm':
            self.normals.append(nn.BatchNorm2d(args.num_filters[0]))
        elif self.normalization == 'DualBatchNorm':
            self.normals.append(DualBatchNorm2d(args.num_filters[0]))
        for n in range(len(args.num_filters)-1):
            self.convs.append(nn.Conv2d(args.num_filters[n], args.num_filters[n+1], kernel_size=args.kernel_sizes[n+1],padding = args.kernel_sizes[n+1]//2 if args.padding else 0))
            if self.normalization == 'GroupNorm':
                self.normals.append(nn.GroupNorm(math.ceil(args.num_filters[n+1]/4),args.num_filters[n+1]))
            elif self.normalization == 'BatchNorm':
                self.normals.append(nn.BatchNorm2d(args.num_filters[n+1]))
            elif self.normalization == 'DualBatchNorm':
                self.normals.append(DualBatchNorm2d(args.num_filters[n+1]))
            if n != len(args.num_filters)-2 or final_pool:
                self.max_pools.append(nn.MaxPool2d(2))
            else:
                self.max_pools.append(None) 
        
        if len(args.mlp_layers)>0:
            self.fcs.append(nn.Linear(conv_out_length, args.mlp_layers[0]))
            if self.normalization == 'BatchNorm':
                self.normals.append(nn.BatchNorm1d(args.mlp_layers[0]))
            elif self.normalization == 'DualBatchNorm':
                self.normals.append(DualBatchNorm1d(args.mlp_layers[0]))
            for n in range(len(args.mlp_layers)-1):
                self.fcs.append(nn.Linear(args.mlp_layers[n], args.mlp_layers[n+1]))
                if self.normalization == 'BatchNorm':
                    self.normals.append(nn.BatchNorm1d(args.mlp_layers[n+1]))
                elif self.normalization == 'DualBatchNorm':
                    self.normals.append(DualBatchNorm1d(args.mlp_layers[n+1]))
            self.fcs.append(nn.Linear(args.mlp_layers[-1], args.num_classes))
        else:
            self.fcs.append(nn.Linear(conv_out_length, args.num_classes))
        
        self.convs = nn.ModuleList(self.convs)
        self.max_pools = nn.ModuleList(self.max_pools)
        self.fcs = nn.ModuleList(self.fcs)
        if self.normalization is not None:
            self.normals = nn.ModuleList(self.normals)

        self.depth = len(self.convs)+len(self.fcs)
        if args.xavier_init:
            for m in self.modules():
                if isinstance(m,nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)

              


class VGG(FedModule):
    def __init__(self, input_shape = [3,32,32],depth = 11,normalization=None,input_normalize_mean = None,input_normalize_var = None,xavier_init=True,num_classes=10):
        super().__init__(normalization,input_normalize_mean,input_normalize_var)
        
        self.input_shape = input_shape


        if depth == 11:# VGG 11
            conv_out_length = (input_shape[1]//32)*(input_shape[2]//32)*512

            self.convs.append(nn.Conv2d(input_shape[0], 64, kernel_size=3,padding = 1))
            self.convs.append(nn.Conv2d(64, 128, kernel_size=3,padding = 1))
            self.convs.append(nn.Conv2d(128, 256, kernel_size=3,padding = 1))
            self.convs.append(nn.Conv2d(256, 256, kernel_size=3,padding = 1))
            self.convs.append(nn.Conv2d(256, 512, kernel_size=3,padding = 1))
            self.convs.append(nn.Conv2d(512, 512, kernel_size=3,padding = 1))
            self.convs.append(nn.Conv2d(512, 512, kernel_size=3,padding = 1))
            self.convs.append(nn.Conv2d(512, 512, kernel_size=3,padding = 1))

            self.max_pools.append(nn.MaxPool2d(2))
            self.max_pools.append(nn.MaxPool2d(2))
            self.max_pools.append(None)
            self.max_pools.append(nn.MaxPool2d(2))
            self.max_pools.append(None)
            self.max_pools.append(nn.MaxPool2d(2))
            self.max_pools.append(None)
            self.max_pools.append(nn.MaxPool2d(2))

            self.fcs.append(nn.Linear(conv_out_length, 512))
            self.fcs.append(nn.Linear(512, 512))
            self.fcs.append(nn.Linear(512, num_classes))

            if self.normalization == 'GroupNorm':
                self.normals.append(nn.GroupNorm(4,64))
                self.normals.append(nn.GroupNorm(8,128))
                self.normals.append(nn.GroupNorm(16,256))
                self.normals.append(nn.GroupNorm(16,256))
                self.normals.append(nn.GroupNorm(32,512))
                self.normals.append(nn.GroupNorm(32,512))
                self.normals.append(nn.GroupNorm(32,512))
                self.normals.append(nn.GroupNorm(32,512))
            elif self.normalization == 'BatchNorm':
                self.normals.append(nn.BatchNorm2d(64))
                self.normals.append(nn.BatchNorm2d(128))
                self.normals.append(nn.BatchNorm2d(256))
                self.normals.append(nn.BatchNorm2d(256))
                self.normals.append(nn.BatchNorm2d(512))
                self.normals.append(nn.BatchNorm2d(512))
                self.normals.append(nn.BatchNorm2d(512))
                self.normals.append(nn.BatchNorm2d(512))
                self.normals.append(nn.BatchNorm1d(512))
                self.normals.append(nn.BatchNorm1d(512))
            elif self.normalization == 'DualBatchNorm':
                self.normals.append(DualBatchNorm2d(64))
                self.normals.append(DualBatchNorm2d(128))
                self.normals.append(DualBatchNorm2d(256))
                self.normals.append(DualBatchNorm2d(256))
                self.normals.append(DualBatchNorm2d(512))
                self.normals.append(DualBatchNorm2d(512))
                self.normals.append(DualBatchNorm2d(512))
                self.normals.append(DualBatchNorm2d(512))
                self.normals.append(DualBatchNorm1d(512))
                self.normals.append(DualBatchNorm1d(512))
        
        elif depth == 9:# VGG 9 Mini
            conv_out_length = (input_shape[1]//16)*(input_shape[2]//16)*64

            self.convs.append(nn.Conv2d(input_shape[0], 8, kernel_size=3,padding = 1))
            self.convs.append(nn.Conv2d(8, 16, kernel_size=3,padding = 1))
            self.convs.append(nn.Conv2d(16, 32, kernel_size=3,padding = 1))
            self.convs.append(nn.Conv2d(32, 32, kernel_size=3,padding = 1))
            self.convs.append(nn.Conv2d(32, 64, kernel_size=3,padding = 1))
            self.convs.append(nn.Conv2d(64, 64, kernel_size=3,padding = 1))
            self.max_pools.append(nn.MaxPool2d(2))
            self.max_pools.append(nn.MaxPool2d(2))
            self.max_pools.append(None)
            self.max_pools.append(nn.MaxPool2d(2))
            self.max_pools.append(None)
            self.max_pools.append(nn.MaxPool2d(2))

            
            self.fcs.append(nn.Linear(conv_out_length, 64))
            self.fcs.append(nn.Linear(64, 64))
            self.fcs.append(nn.Linear(64, num_classes))

            if self.normalization == 'GroupNorm':
                self.normals.append(nn.GroupNorm(1,8))
                self.normals.append(nn.GroupNorm(1,16))
                self.normals.append(nn.GroupNorm(2,32))
                self.normals.append(nn.GroupNorm(2,32))
                self.normals.append(nn.GroupNorm(4,64))
                self.normals.append(nn.GroupNorm(4,64))
            elif self.normalization == 'BatchNorm':
                self.normals.append(nn.BatchNorm2d(8))
                self.normals.append(nn.BatchNorm2d(16))
                self.normals.append(nn.BatchNorm2d(32))
                self.normals.append(nn.BatchNorm2d(32))
                self.normals.append(nn.BatchNorm2d(64))
                self.normals.append(nn.BatchNorm2d(64))
                self.normals.append(nn.BatchNorm1d(64))
                self.normals.append(nn.BatchNorm1d(64))
            elif self.normalization == 'DualBatchNorm':
                self.normals.append(DualBatchNorm2d(8))
                self.normals.append(DualBatchNorm2d(16))
                self.normals.append(DualBatchNorm2d(32))
                self.normals.append(DualBatchNorm2d(32))
                self.normals.append(DualBatchNorm2d(64))
                self.normals.append(DualBatchNorm2d(64))
                self.normals.append(DualBatchNorm1d(64))
                self.normals.append(DualBatchNorm1d(64))
        
        self.convs = nn.ModuleList(self.convs)
        self.max_pools = nn.ModuleList(self.max_pools)
        self.fcs = nn.ModuleList(self.fcs)
        if self.normalization is not None:
            self.normals = nn.ModuleList(self.normals)
        self.depth = len(self.convs)+len(self.fcs)
        
        if xavier_init:
            for m in self.modules():
                if isinstance(m,nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
 


class SmallCNN(FedModule):
    def __init__(self, input_shape = [3,32,32],normalization=None,input_normalize_mean = None,input_normalize_var = None,xavier_init=True,num_classes=10):
        super().__init__(normalization,input_normalize_mean,input_normalize_var)

        self.input_shape = input_shape
        
        conv_out_length = (input_shape[1]//8-2)*(input_shape[2]//8-2)*64

        self.convs.append(nn.Conv2d(input_shape[0], 8, kernel_size=3,padding = 1))
        self.convs.append(nn.Conv2d(8, 16, kernel_size=3,padding = 1))
        self.convs.append(nn.Conv2d(16, 32, kernel_size=3,padding = 1))
        self.convs.append(nn.Conv2d(32, 32, kernel_size=3,padding = 1))
        self.convs.append(nn.Conv2d(32, 64, kernel_size=3))

        self.fcs.append(nn.Linear(conv_out_length, 64))
        self.fcs.append(nn.Linear(64, num_classes))


        self.max_pools.append(nn.MaxPool2d(2))
        self.max_pools.append(nn.MaxPool2d(2))
        self.max_pools.append(None)
        self.max_pools.append(nn.MaxPool2d(2))
        self.max_pools.append(None)


        if self.normalization == 'GroupNorm':
            self.normals.append(nn.GroupNorm(1,8))
            self.normals.append(nn.GroupNorm(2,16))
            self.normals.append(nn.GroupNorm(4,32))
            self.normals.append(nn.GroupNorm(4,32))
            self.normals.append(nn.GroupNorm(8,64))

        elif self.normalization == 'BatchNorm':
            self.normals.append(nn.BatchNorm2d(8))
            self.normals.append(nn.BatchNorm2d(16))
            self.normals.append(nn.BatchNorm2d(32))
            self.normals.append(nn.BatchNorm2d(32))
            self.normals.append(nn.BatchNorm2d(64))
            self.normals.append(nn.BatchNorm1d(64))

        elif self.normalization == 'DualBatchNorm':
            self.normals.append(DualBatchNorm2d(8))
            self.normals.append(DualBatchNorm2d(16))
            self.normals.append(DualBatchNorm2d(32))
            self.normals.append(DualBatchNorm2d(32))
            self.normals.append(DualBatchNorm2d(64))
            self.normals.append(DualBatchNorm1d(64))

        
        
        
        self.convs = nn.ModuleList(self.convs)
        self.max_pools = nn.ModuleList(self.max_pools)
        self.fcs = nn.ModuleList(self.fcs)
        if self.normalization is not None:
            self.normals = nn.ModuleList(self.normals)
        self.depth = len(self.convs)+len(self.fcs)

        if xavier_init:
            for m in self.modules():
                if isinstance(m,nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
        

class early_exit_linear_classifier(nn.Module):
    def __init__(self,input_shape,num_classes,pre_pooling=None,layer=None):
        super().__init__()
        self.input_shape = np.array(input_shape)
        self.pre_pooling = pre_pooling
        input_dim = 1
        if pre_pooling is not None:
            self.pool=nn.MaxPool2d(pre_pooling)
            self.input_shape[2] = self.input_shape[2]//self.pre_pooling
            self.input_shape[3] = self.input_shape[3]//self.pre_pooling 
        for i in self.input_shape[1:]:
            input_dim*=i
        
            #self.pool=nn.AvgPool2d(pre_pooling)
        if layer is not None:
            self.classifier = nn.Sequential(nn.Linear(input_dim,layer),nn.ReLU(),nn.Linear(layer,num_classes))
        else:
            self.classifier = nn.Linear(input_dim,num_classes)
    
    def forward(self,x,*args,**kargs):
        if self.pre_pooling is not None:
            x = self.pool(x)
        x = x.view([x.shape[0],-1])
        return self.classifier(x)

    def reinitialization(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

class early_exit_conv_classifier(nn.Module):
    def __init__(self,input_shape,kernel_size,num_filter,num_classes,xavier_init = False):
        super().__init__()
        self.conv = nn.Conv2d(input_shape[1],num_filter,kernel_size)
        linear_input_dim = num_filter*((input_shape[2]-kernel_size+1)//2)*((input_shape[3]-kernel_size+1)//2)
        # linear_input_dim = num_filter*((input_shape[2]-kernel_size+1))*((input_shape[3]-kernel_size+1))
        self.classifier = nn.Linear(linear_input_dim,num_classes)
        if xavier_init:
            nn.init.xavier_normal_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)
    
    def forward(self,x,*args,**kargs):
        x = F.relu(F.max_pool2d(self.conv(x),2))
        # x = F.relu(self.conv(x)) # no non-linear layers in early classifier
        x = x.view([x.shape[0],-1])
        return self.classifier(x)
    
    def reinitialization(self):
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)


# ==================================ResNet==================================
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, normalization='BatchNorm'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if normalization == 'BatchNorm' else DualBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if normalization == 'BatchNorm' else DualBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.normalization=normalization

    def forward(self, x, adv=False):
        residual = x

        out = self.conv1(x)
        if "Dual" not in self.normalization:
            out = self.bn1(out)
        else:
            out = self.bn1(out,adv)
        
        out = self.relu(out)

        out = self.conv2(out)
        if "Dual" not in self.normalization:
            out = self.bn2(out)
        else:
            out = self.bn2(out,adv)

        if self.downsample is not None:
            if "Dual" not in self.normalization:
                residual = self.downsample(x)
            else:
                residual = self.downsample(x,adv)

        out += residual
        out = self.relu(out)

        return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         #out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out

class DBNSequential(nn.Sequential):
    def __init__(self,*args):
        super(DBNSequential,self).__init__(*args)

    def forward(self,x,adv=False):
        for module in self:
            if isinstance(module,DualBatchNorm2d) or isinstance(module,DualBatchNorm1d):
                x = module(x,adv)
            else:
                x = module(x)
        return x

class ResNet(FedModule):

    def __init__(self, input_shape=[3,32,32],depth=20, num_classes=10, normalization='BatchNorm',input_normalize_mean = None,input_normalize_var = None):
        super(ResNet, self).__init__(normalization=normalization,input_normalize_mean = input_normalize_mean,input_normalize_var = input_normalize_var)
        # Model type specifies number of layers for CIFAR-10 model
        self.input_shape = input_shape
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        

        #block = Bottleneck if depth >=44 else BasicBlock
        block = BasicBlock
        
        self.inplanes = 16
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16) if self.normalization == 'BatchNorm' else DualBatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        layers = []
        layers.extend(self._make_layers(block, 16, n))
        layers.extend(self._make_layers(block, 32, n, stride=2))
        layers.extend(self._make_layers(block, 64, n, stride=2))
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.depth = len(self.layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m,DualBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layers(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.normalization == 'BatchNorm':
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = DBNSequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                    DualBatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,normalization=self.normalization))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,normalization=self.normalization))

        return layers

    def forward(self, x, enter_stage=0, early_exit=None, early_stop=False, adv=False):
        if self.input_normalize is not None:
            x = self.input_normalize(x)
        if enter_stage is None:
            enter_stage = 0
        int_exit = isinstance(early_exit,int)
        if int_exit:
            early_exit=[early_exit,]

        stage = 0
        early_exit_output=[]

        if x.dim() == 4:
            if enter_stage == 0:
                x = self.conv(x)
                if "Dual" not in self.normalization:
                    x = self.bn(x)
                else:
                    x = self.bn(x,adv)
                x = self.relu(x)  # 32x32
            for n in range(len(self.layers)):
                if stage >= enter_stage:
                    if "Dual" not in self.normalization:
                        x = self.layers[n](x)
                    else:
                        x = self.layers[n](x,adv)
                stage += 1
                if early_exit is not None and stage in early_exit:
                    early_exit_output.append(x.clone())
                    if early_stop:
                        return early_exit_output[-1]
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        x = self.fc(x)
        if early_exit is not None:
            return x, early_exit_output if not int_exit else early_exit_output[0]
        return x

    def Get_Local_State_Dict(self):
        sd = self.state_dict()
        for name in list(sd.keys()):
            if 'bn' not in name and 'downsample.1' not in name:
                sd.pop(name)
        return sd


    

    def get_layer_state_dict(self, layers, normalization_params=False):
        # Return the state dict with None on the other layers
        layer_state_dict = {k: None for k in self.state_dict().keys()}
        for i in layers:
            if i == 0:# include the first conv layer
                layer_state_dict['conv.weight'] = self.state_dict()['conv.weight']
                if normalization_params:
                    layer_state_dict['bn.weight'] = self.state_dict()['bn.weight']
                    layer_state_dict['bn.bias'] = self.state_dict()['bn.bias']
                    layer_state_dict['bn.running_mean'] = self.state_dict()['bn.running_mean']
                    layer_state_dict['bn.running_var'] = self.state_dict()['bn.running_var']
                    layer_state_dict['bn.num_batches_tracked'] = self.state_dict()['bn.num_batches_tracked']
                    if self.normalization == 'DualBatchNorm':
                        layer_state_dict['bn.adv_running_mean'] = self.state_dict()['bn.adv_running_mean']
                        layer_state_dict['bn.adv_running_var'] = self.state_dict()['bn.adv_running_var']
                        layer_state_dict['bn.adv_num_batches_tracked'] = self.state_dict()['bn.adv_num_batches_tracked']
            elif i == len(self.layers) - 1: # include the last fc layer
                layer_state_dict['fc.weight'] = self.state_dict()['fc.weight']
                layer_state_dict['fc.bias'] = self.state_dict()['fc.bias']
                
            layer_state_dict[f'layers.{i}.conv1.weight'] = self.state_dict()[f'layers.{i}.conv1.weight']
            layer_state_dict[f'layers.{i}.conv2.weight'] = self.state_dict()[f'layers.{i}.conv2.weight']
            if f'layers.{i}.downsample.0.weight' in self.state_dict():
                layer_state_dict[f'layers.{i}.downsample.0.weight'] = self.state_dict()[f'layers.{i}.downsample.0.weight']
            
            if normalization_params:
                layer_state_dict[f'layers.{i}.bn1.weight'] = self.state_dict()[f'layers.{i}.bn1.weight']
                layer_state_dict[f'layers.{i}.bn1.bias'] = self.state_dict()[f'layers.{i}.bn1.bias']
                layer_state_dict[f'layers.{i}.bn1.running_mean'] = self.state_dict()[f'layers.{i}.bn1.running_mean']
                layer_state_dict[f'layers.{i}.bn1.running_var'] = self.state_dict()[f'layers.{i}.bn1.running_var']
                layer_state_dict[f'layers.{i}.bn1.num_batches_tracked'] = self.state_dict()[f'layers.{i}.bn1.num_batches_tracked']
                layer_state_dict[f'layers.{i}.bn2.weight'] = self.state_dict()[f'layers.{i}.bn2.weight']
                layer_state_dict[f'layers.{i}.bn2.bias'] = self.state_dict()[f'layers.{i}.bn2.bias']
                layer_state_dict[f'layers.{i}.bn2.running_mean'] = self.state_dict()[f'layers.{i}.bn2.running_mean']
                layer_state_dict[f'layers.{i}.bn2.running_var'] = self.state_dict()[f'layers.{i}.bn2.running_var']
                layer_state_dict[f'layers.{i}.bn2.num_batches_tracked'] = self.state_dict()[f'layers.{i}.bn2.num_batches_tracked']
                if self.normalization == 'DualBatchNorm':
                    layer_state_dict[f'layers.{i}.bn1.adv_running_mean'] = self.state_dict()[f'layers.{i}.bn1.adv_running_mean']
                    layer_state_dict[f'layers.{i}.bn1.adv_running_var'] = self.state_dict()[f'layers.{i}.bn1.adv_running_var']
                    layer_state_dict[f'layers.{i}.bn1.adv_num_batches_tracked'] = self.state_dict()[f'layers.{i}.bn1.adv_num_batches_tracked']
                    layer_state_dict[f'layers.{i}.bn2.adv_running_mean'] = self.state_dict()[f'layers.{i}.bn2.adv_running_mean']
                    layer_state_dict[f'layers.{i}.bn2.adv_running_var'] = self.state_dict()[f'layers.{i}.bn2.adv_running_var']
                    layer_state_dict[f'layers.{i}.bn2.adv_num_batches_tracked'] = self.state_dict()[f'layers.{i}.bn2.adv_num_batches_tracked']
                if f'layers.{i}.downsample.1.weight' in self.state_dict():
                    layer_state_dict[f'layers.{i}.downsample.1.weight'] = self.state_dict()[f'layers.{i}.downsample.1.weight']
                    layer_state_dict[f'layers.{i}.downsample.1.bias'] = self.state_dict()[f'layers.{i}.downsample.1.bias']
                    layer_state_dict[f'layers.{i}.downsample.1.running_mean'] = self.state_dict()[f'layers.{i}.downsample.1.running_mean']
                    layer_state_dict[f'layers.{i}.downsample.1.running_var'] = self.state_dict()[f'layers.{i}.downsample.1.running_var']
                    layer_state_dict[f'layers.{i}.downsample.1.num_batches_tracked'] = self.state_dict()[f'layers.{i}.downsample.1.num_batches_tracked']
                    if self.normalization == 'DualBatchNorm':
                        layer_state_dict[f'layers.{i}.downsample.1.adv_running_mean'] = self.state_dict()[f'layers.{i}.downsample.1.adv_running_mean']
                        layer_state_dict[f'layers.{i}.downsample.1.adv_running_var'] = self.state_dict()[f'layers.{i}.downsample.1.adv_running_var']
                        layer_state_dict[f'layers.{i}.downsample.1.adv_num_batches_tracked'] = self.state_dict()[f'layers.{i}.downsample.1.adv_num_batches_tracked']

        return layer_state_dict

    def get_early_exit_shape(self,exit_layer):
        test = torch.zeros(self.input_shape).unsqueeze(0).to(self.fc.weight)
        output = self.forward(test,early_exit=exit_layer,early_stop=True)
        return output.shape

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

# if __name__ == '__main__':
#     model = get_net('lenet5','mnist',10,modularization=True,adv_norm=True)
#     for name,_ in model.named_modules():
#         print(name)
#     print(model.feature_layer_list)
#     print(model.module_list)
    