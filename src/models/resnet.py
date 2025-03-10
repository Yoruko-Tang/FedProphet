import torch
import torch.nn as nn
from typing import List
import types
import numpy as np

from torchvision.models.resnet import _resnet,BasicBlock
from models.DBN import DualBatchNorm2d,adv,clean

def resnet10(**kwargs):
    return _resnet(BasicBlock, [1, 1, 1, 1],weights=None,progress=False)

def adapt(model,modeltype,num_classes):
    """
    Adapt the network with given modeltype and number of classes
    """
    #if num_classes!=1000: # reinitialize the last layer
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    model.output_size = num_classes
    return model

def add_normalization(model,normalization_layer):
    """
    Add normalization layer at the beginning of a model for adversarial training
    """
    model.normalize = normalization_layer
    # add this layer into the forward function
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        x = self._forward_impl(x)
        return x
    
    model.forward = types.MethodType(forward,model)
    return model

def replace_norm(model, norm='BN'):
    if norm == 'LN':
        model._norm_layer = nn.LayerNorm
    elif norm == 'GN':
        model._norm_layer = nn.GroupNorm
    elif norm == 'IN':
        model._norm_layer = nn.InstanceNorm2d
    elif norm == 'sBN':
        model._norm_layer = nn.BatchNorm2d
    elif norm == 'BN':
        model._norm_layer = nn.BatchNorm2d
        return model
    elif norm == 'DBN':
        model._norm_layer = DualBatchNorm2d
        model.adv = types.MethodType(adv,model)
        model.clean = types.MethodType(clean,model)
    else: # remove the normalization
        model._norm_layer = nn.Identity
    
    for n,m in model.named_modules():
        if isinstance(m,nn.BatchNorm2d):
            if norm == 'LN':
                norm_layer = nn.LayerNorm(m.num_features)
            elif norm == 'GN':
                norm_layer = nn.GroupNorm(4,m.num_features)
            elif norm == 'IN':
                norm_layer == nn.InstanceNorm2d(m.num_features)
            elif norm == 'sBN':
                norm_layer = nn.BatchNorm2d(m.num_features,track_running_stats=False)
                # initialize the stat with the current clean stat
                norm_layer.running_mean.copy_(m.running_mean) 
                norm_layer.running_var.copy_(m.running_var)
                norm_layer.num_batches_tracked.copy_(m.num_batches_tracked)
                norm_layer.weight.data.copy_(m.weight.data)
                norm_layer.bias.data.copy_(m.bias.data)
            elif norm == 'DBN':
                norm_layer = DualBatchNorm2d(m.num_features)
                # initialize the adv stat with the current clean stat
                norm_layer.running_mean.copy_(m.running_mean)
                norm_layer.adv_running_mean.copy_(m.running_mean)
                norm_layer.running_var.copy_(m.running_var)
                norm_layer.adv_running_var.copy_(m.running_var)
                norm_layer.num_batches_tracked.copy_(m.num_batches_tracked)
                norm_layer.adv_num_batches_tracked.copy_(m.num_batches_tracked)
                norm_layer.weight.data.copy_(m.weight.data)
                norm_layer.bias.data.copy_(m.bias.data)
            else: # remove the normalization
                norm_layer = nn.Identity()
            norm_layer.num_features = m.num_features
            layer_name = n.split(".")[0]
            if "bn" in layer_name:
                setattr(model,n,norm_layer)
            else:
                block_idx = int(n.split(".")[1])
                block = getattr(model,layer_name)[block_idx]
                block_mod_name = n.split(".")[2]
                if "downsample" in block_mod_name:
                    bn_idx = int(n.split(".")[3])
                    block.downsample[bn_idx] = norm_layer
                else:
                    setattr(block,block_mod_name,norm_layer)
    return model

def modularization(model):
    """
    Make the model into cascaded modules.

    This function will add a module_forward function for a resnet model,
    such that the forward only go through layers in a given list.
    If the given list does not contain contious layers, there may be an error.
    [Warning]: There is no check on the continuity of the list!

    This function will also set the granulity of the model, which is the atom in modularization
    """

    def module_forward(self, x: torch.Tensor, module_list: List[str]) -> torch.Tensor:
        layer_list = []
        for m in module_list:
            layer_list += m.split("+")
        valid_features = []
        valid_classifier = []
        all_modules = {}
        for n,m in self.named_modules():
            all_modules[n] = m
        for n in layer_list:
            if 'fc' not in n:
                valid_features.append(all_modules[n])
    
            else:
                valid_classifier.append(all_modules[n])
        
        if len(valid_features)>0:
            for f in valid_features:
                x = f(x)
                
        if len(valid_classifier)>0:
            if x.dim()>2:
                x = torch.flatten(x, 1)
            for f in valid_classifier:
                x = f(x)

        return x
    
    model.module_forward = types.MethodType(module_forward,model)
    
    module_list = []
    continue_layer = []
    for n,m in model.named_modules():
        if n.count('.') == 0 and n != '' and 'layer' not in n:
            if n == 'normalize':
                module_list.insert(0,n)
            elif isinstance(m,nn.Conv2d): # 'conv1'
                if len(continue_layer)>0:
                    module_list.append(continue_layer)
                continue_layer = [n]
            else: # 'bn1', 'relu', 'maxpool', 'avgpool', 'fc'
                continue_layer += [n]
        elif n.count('.') == 1: # 'layerx.x'
            if len(continue_layer)>0:
                module_list.append(continue_layer)
            continue_layer = [n]

    if len(continue_layer) > 0: # the last continue layer has not been appended
        module_list.append(continue_layer)

    model.module_list = module_list
    return model


def partialization(model):
    """
    This function will modify the model to enable partial training.
    """
    
    
    def partial_forward(self,x: torch.Tensor, neuron_dict: dict) -> torch.Tensor:
        def partial_forward_hook(module,fea_in,fea_out):
            if isinstance(module,(nn.Conv2d,nn.Linear)):
                # for linear and conv layers
                # do not mask the output for now, but only scale the output
                fea_out = fea_out/(len(module.in_retain_idx)/fea_in[0].size(1))
                if hasattr(module,'retain_idx'):
                    return [fea_out,module.retain_idx] # tell the next module the retained idx
                else: # for the last layer of the model
                    return fea_out
            else:
                # for normalization layer and relu layer, mask the output
                mask = torch.zeros_like(fea_out)
                mask[:,module.retain_idx] = 1.0
                fea_out = fea_out*mask
                return [fea_out,module.retain_idx]

        
        def partial_pre_forward_hook(module,fea_in):
            if len(fea_in[0]) == 4: # shortcut output
                assert all(fea_in[0][1] == fea_in[0][3]), "The retain channel of the shortcut must be the same as that of the last conv layer!"
                
                fea_in = [[fea_in[0][0]+fea_in[0][2],fea_in[0][1]]]
            if isinstance(module,(nn.Conv2d,nn.Linear)):
                module.in_retain_idx = fea_in[0][1] # record the input mask
            else:
                module.retain_idx = fea_in[0][1] # record the output mask
            return fea_in[0][0]

        handles = []
        pre_handles = []
        for n,m in self.named_modules():
            if n.count('.') == 3 or (n.count('.') == 2 and 'downsample' not in n) or (n.count('.') == 0 and n != '' and 'layer' not in n):
                # register retain_idx
                if n in neuron_dict and n in self.neuron_num:
                    if 'conv3' in n and n.replace('conv3','downsample.0') not in neuron_dict: # bottleneck
                        first_block = n.split('.')
                        if first_block[1] == '0': # a layer without downsampling at the first bottleneck block
                            # use the output idx of the last layer for this block
                            if first_block[0] == 'layer1':
                                previous_layer = 'conv1'
                            else:
                                previous_layer = 'layer{}.0.conv3'.format(int(first_block[0][-1])-1)
                            neuron_dict[n] = neuron_dict[previous_layer]
                            m.retain_idx = neuron_dict[n]
                        else:
                            first_block[1] = '0'
                            first_block = '.'.join(first_block)
                            m.retain_idx = neuron_dict[first_block]
                    elif 'conv2' in n and n.replace('conv2','downsample.0') not in neuron_dict and n.replace('conv2','conv3') not in neuron_dict: # basicblock
                        first_block = n.split('.')
                        if first_block[1] == '0': # a layer without downsampling at the first basic block
                            # use the output idx of the last layer for this block
                            if first_block[0] == 'layer1':
                                previous_layer = 'conv1'
                            else:
                                previous_layer = 'layer{}.0.conv2'.format(int(first_block[0][-1])-1)
                            neuron_dict[n] = neuron_dict[previous_layer]
                            m.retain_idx = neuron_dict[n]
                        else:
                            first_block[1] = '0'
                            first_block = '.'.join(first_block)
                            m.retain_idx = neuron_dict[first_block]
                    elif 'downsample.0' in n:
                        if n.replace('downsample.0','conv3') in neuron_dict: # bottleneck
                            m.retain_idx = neuron_dict[n.replace('downsample.0','conv3')]
                        else: # basicblock
                            m.retain_idx = neuron_dict[n.replace('downsample.0','conv2')]
                    else:
                        m.retain_idx = neuron_dict[n]
                
                # register in_retain_idx
                if n != 'fc':
                    ph = m.register_forward_pre_hook(partial_pre_forward_hook)
                else: # the first linear layer
                    last_feature_layer = "layer4.0.conv3"
                    if last_feature_layer not in neuron_dict: # basic block
                        last_feature_layer = "layer4.0.conv2"
                    m.in_retain_idx = np.arange(m.in_features).reshape((-1,) + self.avgpool.output_size)
                    m.in_retain_idx = m.in_retain_idx[neuron_dict[last_feature_layer]].flatten()

                # register retain_idx propagate
                if n != 'avgpool':
                    h = m.register_forward_hook(partial_forward_hook)
                
                handles.append(h)
                pre_handles.append(ph)
                
        x = self.forward([x,np.arange(x.size(1))])

        for h in handles:
            h.remove() # remove the hook
        for ph in pre_handles:
            ph.remove()

        return x
    
    
    neuron_num = {}
    for n,m in model.named_modules():
        
        if isinstance(m,nn.Conv2d):
            neuron_num[n] = m.out_channels
        elif isinstance(m,nn.Linear) and n!='fc':
            neuron_num[n] = m.out_features
    
    
    model.partial_forward = types.MethodType(partial_forward,model)
    model.neuron_num = neuron_num

    return model


def set_feature_layer(model):
    """
    Set the layers whose input must be preserved in the memory for backpropagation 
    """
    feature_layer_list = []
    for n in model.state_dict().keys():
        if '.weight' in n and 'downsample.0' not in n:
            feature_layer_list.append(n.replace('.weight',''))
    model.feature_layer_list = feature_layer_list
    return model

def set_representation_layer(model):
    """
    Set the representation layer for FedET 
    """
    model.rep_layers = ["fc.weight","fc.bias"]
    return model
    