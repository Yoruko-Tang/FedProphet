import torch
import torch.nn as nn
import types
from typing import List

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

def lenet5(**kwargs):
    return LeNet5()

def adapt(model,modeltype,num_classes):
    """
    Adapt the network with given modeltype and number of classes
    """
    #if num_classes!=1000: # reinitialize the last layer
    assert modeltype == 'mnist', "LeNet5 can only be applied on MNIST-like datasets!"
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feat, num_classes)
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

def modularization(model):
    """
    Make the model into cascaded modules.

    This function will add a module_forward function for a resnet model,
    such that the forward only go through layers in a given list.
    If the given list does not contain contious layers, there may be an error.
    [Warning]: There is no check on the continuity of the list!
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
            if 'features.5' in layer_list:
                x = torch.flatten(x, 1)
        if len(valid_classifier)>0:
            for f in valid_classifier:
                x = f(x)

        return x
    
    model.module_forward = types.MethodType(module_forward,model)
    return model