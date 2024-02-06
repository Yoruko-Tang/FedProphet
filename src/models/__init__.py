# regisiter your models here to be considered
from torchvision.models.vgg import *
from torchvision.models.resnet import *
from models.lenet import lenet5

import models.resnet as resnet
import models.vgg as vgg
import models.lenet as lenet

def modelname_to_modelfamily(modelname):
    if 'vgg' in modelname:
        modelfamily = 'vgg'
    elif 'resnet' in modelname:
        modelfamily = 'resnet'
    elif 'lenet' in modelname:
        modelfamily = 'lenet'
    else:
        raise RuntimeError("Not supported model: "+modelname)

    return modelfamily
