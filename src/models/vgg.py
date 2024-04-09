import torch
import torch.nn as nn
from typing import List
import types
import numpy as np

def adapt(model,modeltype,num_classes):
    """
    Adapt the network with given modeltype and number of classes
    """
    #if num_classes!=1000: # reinitialize the last layer
    if modeltype == 'imagenet': # for 224x224x3 inputs
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)
    else: # for 32x32x3 inputs
        model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes))# use less neurons for a small input
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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    model.forward = types.MethodType(forward,model)
    return model

def replace_norm(model,norm='BN'):
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
            getattr(model,n.split('.')[0])[int(n.split('.')[1])] = norm_layer
    return model



def modularization(model):
    """
    Make the model into cascaded modules.

    This function will add a module_forward function for a vgg model,
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
            if 'classifier' not in n:
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
        if n.count('.') == 0: # '','normalize', 'features', 'avgpool', 'classifier'
            if n == 'normalize':
                module_list.insert(0,n)
            elif n == 'avgpool':
                continue_layer += [n]
            # elif n == 'classifier':
            #     if len(continue_layer)>0:
            #         module_list.append(continue_layer)
            #     continue_layer = [n]
            
        elif n.count('.') == 1: # 'features.x', 'classifier.x'
            if isinstance(m,nn.Conv2d):
                if len(continue_layer)>0:
                    module_list.append(continue_layer)
                continue_layer = [n]
            else:
                continue_layer += [n]
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
            if isinstance(module,(nn.Conv2d,nn.Linear)):
                module.in_retain_idx = fea_in[0][1] # record the input mask
            else:
                module.retain_idx = fea_in[0][1] # record the output mask
            return fea_in[0][0]

        handles = []
        pre_handles = []
        for n,m in self.named_modules():
            if n not in ["features","","classifier"]:
                # register retain_idx
                if n in neuron_dict and n in self.neuron_num:
                    m.retain_idx = neuron_dict[n]
                
                # register in_retain_idx
                if n!= 'classifier.0':
                    ph = m.register_forward_pre_hook(partial_pre_forward_hook)
                else: # the first linear layer
                    last_feature_layer = "features.%d"%(len(self.features)-4)
                    if last_feature_layer not in neuron_dict: # vgg w/o bn
                        last_feature_layer = "features.%d"%(len(self.features)-3)
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
        elif isinstance(m,nn.Linear) and n!="classifier.%d"%(len(model.classifier)-1):# do not mask the last layer
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
        if '.weight' in n:
            feature_layer_list.append(n.replace('.weight',''))
    model.feature_layer_list = feature_layer_list
    return model

def set_representation_layer(model):
    """
    Set the representation layer for FedET 
    """
    model.rep_layers = ["classifier.6.weight","classifier.6.bias"]
    return model
    

