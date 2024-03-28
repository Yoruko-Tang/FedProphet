# regisiter your models here to be considered
from torchvision.models.vgg import *
from torchvision.models.resnet import *
from models.cnn import lenet5,cnn6

import models.resnet as resnet
import models.vgg as vgg
import models.cnn as cnn

def modelname_to_modelfamily(modelname):
    if 'vgg' in modelname:
        modelfamily = 'vgg'
    elif 'resnet' in modelname:
        modelfamily = 'resnet'
    elif 'lenet' in modelname or 'cnn' in modelname:
        modelfamily = 'cnn'
    else:
        raise RuntimeError("Not supported model: "+modelname)

    return modelfamily
