import os.path as osp

from torchvision.datasets import MNIST as TVMNIST
from torchvision.datasets import FashionMNIST as TVFashionMNIST
from torchvision.datasets import KMNIST as TVKMNIST




class MNIST(TVMNIST):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join('./data', 'mnist')
        super().__init__(root, train, transform, target_transform, download)


class KMNIST(TVKMNIST):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join('./data', 'kmnist')
        super().__init__(root, train, transform, target_transform, download)


class FashionMNIST(TVFashionMNIST):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join('./data', 'mnist_fashion')
        super().__init__(root, train, transform, target_transform, download)
