import torch
import torch.nn as nn
from typing import List
import types

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
            mask = torch.zeros_like(fea_out)
            mask[:,module.retain_idx] = 1.0
            fea_out = fea_out*mask/(len(module.retain_idx)/fea_out.size(1))
            return fea_out
        handles = []
        for n,m in self.named_modules():
            if n in neuron_dict and n in self.neuron_num:
                m.retain_idx = neuron_dict[n]
                h = m.register_forward_hook(partial_forward_hook)
                handles.append(h)
                
        x = self.forward(x)

        for h in handles:
            h.remove() # remove the hook

        return x
    
    
    
    neuron_num = {}
    for n,m in model.named_modules():
        if isinstance(m,nn.Conv2d):
            neuron_num[n] = m.out_channels
            m.retain_idx = None
        # elif isinstance(m,nn.Linear):
        #     neuron_num[n] = m.out_features
        #     m.retain_idx = None
    
    model.partial_forward = types.MethodType(partial_forward,model)
    model.neuron_num = neuron_num

    return model


def set_feature_layer(model):
    """
    Set the layers whose input must be preserved in the memory for backpropagation 
    """
    feature_layer_list = []
    for n in model.state_dict().keys():
        if '.weight' in n:
            feature_layer_list.append(n.replace('.weight',''))
    model.feature_layer_list = feature_layer_list
    return model

def set_representation_layer(model):
    """
    Set the representation layer for FedET 
    """
    model.rep_layers = ["fc.weight","fc.bias"]
    return model
    