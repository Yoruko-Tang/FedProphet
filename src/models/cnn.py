import torch
import torch.nn as nn
import types
from typing import List
import numpy as np

class LeNet5(nn.Module):
    """
    Lenet5 for mnist-like dataset only
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        features = []
        conv1 = nn.Conv2d(1, 6, 5, padding=2)
        conv2 = nn.Conv2d(6, 16, 5)
        features += [conv1,nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2,stride=2)]
        features += [conv2,nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2,stride=2)]
        self.features = nn.Sequential(*features)
        classifier = []
        fc1   = nn.Linear(16*5*5, 120)
        fc2   = nn.Linear(120, 84)
        fc3   = nn.Linear(84, 10)
        classifier += [fc1,nn.ReLU(inplace=True)]
        classifier += [fc2,nn.ReLU(inplace=True)]
        classifier += [fc3]
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)

        return x
    
class CNN6(nn.Module):
    """
    CNN model for CIFAR-like dataset only
    """
    def __init__(self):
        super().__init__()
        features = []
        conv1 = nn.Conv2d(3, 64, 3, padding=1)
        conv2 = nn.Conv2d(64, 64, 3,padding=1)
        conv3 = nn.Conv2d(64, 128, 3, padding=1)
        conv4 = nn.Conv2d(128, 128, 3,padding=1)
        features += [conv1,nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2,stride=2)] # 16x16x64
        features += [conv2,nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2,stride=2)] # 8x8x64
        features += [conv3,nn.BatchNorm2d(128),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2,stride=2)] # 4x4x128
        features += [conv4,nn.BatchNorm2d(128),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2,stride=2)] # 2x2x128
        self.features = nn.Sequential(*features)
        classifier = []
        fc1   = nn.Linear(2*2*128, 512)
        fc2   = nn.Linear(512, 10)
        classifier += [fc1,nn.ReLU(inplace=True)]
        classifier += [fc2]

        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)

        return x
    
class CNN3(nn.Module):
    """
    CNN model for CIFAR-like dataset only
    """
    def __init__(self):
        super().__init__()
        features = []
        conv1 = nn.Conv2d(3, 64, 3, padding=1)
        conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        features += [conv1,nn.BatchNorm2d(64),nn.ReLU(inplace=True)] # 32x32x64
        features += [conv2,nn.BatchNorm2d(128),nn.ReLU(inplace=True),nn.AdaptiveMaxPool2d([2,2])] # 2x2x128
        self.features = nn.Sequential(*features)
        classifier = []
        fc   = nn.Linear(512, 10)
        classifier += [fc]

        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)

        return x
    
class CNN4(nn.Module):
    """
    CNN model for ImageNet-like dataset only
    """
    def __init__(self):
        super().__init__()
        features = []
        conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        conv2 = nn.Conv2d(64, 128, 7,stride=2, padding=3)
        features += [conv1,nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2,stride=2)] # 56x56x64
        features += [conv2,nn.BatchNorm2d(128),nn.ReLU(inplace=True),nn.AdaptiveMaxPool2d([7,7])] # 7x7x128
        self.features = nn.Sequential(*features)
        classifier = []
        fc1 = nn.Linear(7*7*128,512)
        fc2   = nn.Linear(512, 100)
        classifier += [fc1,nn.ReLU(inplace=True)]
        classifier += [fc2]

        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)

        return x

def lenet5(**kwargs):
    return LeNet5()

def cnn6(**kwargs):
    return CNN6()

def cnn3(**kwargs):
    return CNN3()

def cnn4(**kwargs):
    return CNN4()

def adapt(model,modeltype,num_classes):
    """
    Adapt the network with given modeltype and number of classes
    """
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feat, num_classes)
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
        x = torch.flatten(x,1)
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
        if n == 'normalize':
            module_list.insert(0,n)
        elif n.count('.') == 1:
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
            #print(fea_in)
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
                if n != "classifier.0":
                    ph = m.register_forward_pre_hook(partial_pre_forward_hook)
                else: # the first linear layer
                    last_feature_layer = "features.%d"%(len(self.features)-4)
                    if last_feature_layer not in neuron_dict: # lenet
                        last_feature_layer = "features.%d"%(len(self.features)-3)
                        out_channel = self.features[len(self.features)-3].out_channels
                    else:
                        out_channel = self.features[len(self.features)-4].out_channels
                    out_size = int(np.sqrt(self.classifier[0].in_features/out_channel))
                    m.in_retain_idx = np.arange(m.in_features).reshape((out_channel,out_size,out_size))
                    m.in_retain_idx = m.in_retain_idx[neuron_dict[last_feature_layer]].flatten()
                
                # register retain_idx propagate
                if n != 'features.%d'%(len(self.features)-1):
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
    model.rep_layers = ["classifier.%d.weight"%(len(model.classifier)-1),
                        "classifier.%d.bias"%(len(model.classifier)-1)]
    return model