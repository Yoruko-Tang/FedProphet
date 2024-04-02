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
            pretrained=False, norm_type = "BN", 
            adv_norm=False, modularization=False):
    """
    modelname: must be exactly the same as the classes in torchvision.models
    e.g., vgg11, vgg16
    """
    
    # get pretrained model
    model = eval('models.{}'.format(modelname))(weights="DEFAULT" if pretrained else None)
    # adapt to the specified num_classes
    model = eval('models.{}.adapt'.format(models.modelname_to_modelfamily(modelname)))(model,modeltype,num_classes)
    # replace norm layer
    if norm_type != "BN":
        model = eval('models.{}.replace_norm'.format(models.modelname_to_modelfamily(modelname)))(model,modeltype,norm_type)
    # add list of feature layers for memory tracking
    model = eval('models.{}.set_feature_layer'.format(models.modelname_to_modelfamily(modelname)))(model)
    # add representation layers for FedET
    model = eval('models.{}.set_representation_layer'.format(models.modelname_to_modelfamily(modelname)))(model)


    # add normalization layer to the adversarial training model
    if adv_norm:
        normalization_layer = datafamily_to_normalize[modeltype]
        model = eval('models.{}.add_normalization'.format(models.modelname_to_modelfamily(modelname)))(model,normalization_layer)
    
    # modularize the model such that the model can enter and exit at any layers
    if modularization:
        model = eval('models.{}.modularization'.format(models.modelname_to_modelfamily(modelname)))(model)
    return model
    


class PT_Mask(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    
    def forward(self,x,retain_idx):
        mask = torch.zeros([1,x.shape[1]]+[1]*(x.dim()-2)).to(x)
        mask[0,retain_idx] = 1.0
        return mask*x
