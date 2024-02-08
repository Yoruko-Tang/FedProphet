import torch
import torch.nn as nn
from typing import List
import types

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

    

def modularization(model):
    """
    Make the model into cascaded modules.

    This function will add a module_forward function for a vgg model,
    such that the forward only go through layers in a given list.
    If the given list does not contain contious layers, there may be an error.
    [Warning]: There is no check on the continuity of the list!
    
    This function will also set the granulity of the model, which is the atom in modularization
    """

    def module_forward(self, x: torch.Tensor, layer_list: List[str]) -> torch.Tensor:
        valid_features = []
        valid_classifier = []
        for n,m in self.named_modules():
            if n in layer_list:
                if 'classifier' not in n:
                    valid_features += [m]
        
                else:
                    valid_classifier += [m]
        
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


