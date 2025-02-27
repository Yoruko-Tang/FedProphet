import torch
import torch.nn as nn
import types
from typing import List
import numpy as np



def adapt(model,modeltype,num_classes):
    """
    Adapt the network with given modeltype and number of classes
    """

    in_feat = model.heads.head.in_features
    model.heads.head = nn.Linear(in_feat, num_classes)
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
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x
    
    model.forward = types.MethodType(forward,model)
    return model

def replace_norm(model,norm='LN'):
    """
    ViT does not allow to replace the layer norm
    """
    
    model._norm_layer = nn.LayerNorm
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
        device = x.device
        layer_list = []
        for m in module_list:
            layer_list += m.split("+")
        valid_embedding = False
        valid_features = []
        valid_classifier = None
        normalize = False
        all_modules = {}
        for n,m in self.named_modules():
            all_modules[n] = m
        for n in layer_list:
            if n == "normalize":
                normalize = True
            elif 'conv_proj' in n:
                valid_embedding = True
            elif 'heads' in n:
                valid_classifier = all_modules[n]
            else: # transformer.layers.x, transformer.norm
                valid_features.append(all_modules[n])
        
        if normalize:
            x = self.normalize(x)
        
        if valid_embedding: # embedding part
            # Reshape and permute the input tensor
            x = self._process_input(x)
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            # Add position embedding
            x += self.encoder.pos_embedding

            x = self.encoder.dropout(x)

        if valid_features: # transformer part
            for f in valid_features:
                x = f(x)

            
        if valid_classifier: # classifier part
            x = x[:,0]
            
            x = valid_classifier(x)

        return x
    
    model.module_forward = types.MethodType(module_forward,model)

    module_list = []

    
    for n,m in model.named_modules():
        # encoder.layers.encoder_layer_x
        if ("layers" in n and n.count(".") == 2):
            module_list.append([n])
    pre_process = ["normalize","conv_proj"] if hasattr(model,"normalize") else ["conv_proj"]
    module_list[0] = pre_process + module_list[0]
    module_list[-1] += ["encoder.ln","heads"]
    
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

def set_representation_layer(model):
    """
    Set the representation layer for FedET 
    """
    model.rep_layers = ["heads.head.weight","heads.head.bias"]
    return model

