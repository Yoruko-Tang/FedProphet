# regisiter your models here to be considered
from torchvision.models.vgg import *
from torchvision.models.resnet import *
from torchvision.models.vision_transformer import *
from models.cnn import lenet5,cnn6,cnn4,cnn3
from models.resnet import resnet10


import models.resnet as resnet
import models.vgg as vgg
import models.cnn as cnn
import models.vit as vit

def modelname_to_modelfamily(modelname):
    if 'vgg' in modelname:
        modelfamily = 'vgg'
    elif 'resnet' in modelname:
        modelfamily = 'resnet'
    elif 'lenet' in modelname or 'cnn' in modelname:
        modelfamily = 'cnn'
    elif 'vit' in modelname:
        modelfamily = 'vit'
    else:
        raise RuntimeError("Not supported model: "+modelname)

    return modelfamily
