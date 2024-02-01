# regisiter your models here to be considered
from torchvision.models.vgg import *
from torchvision.models.resnet import *
from models.lenet5 import lenet5

def modelname_to_modelfamily(modelname):
    if 'vgg' in modelname:
        modelfamily = 'vgg'
    elif 'resnet' in modelname:
        modelfamily = 'resnet'
    elif 'lenet5' in modelname:
        modelfamily = 'lenet5'
    else:
        raise RuntimeError("Not supported model: "+modelname)

    return modelfamily
