import os.path as osp

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets import CIFAR10 as TVCIFAR10
from torchvision.datasets import CIFAR100 as TVCIFAR100
from torchvision.datasets import SVHN as TVSVHN



class CIFAR10(TVCIFAR10):
    base_folder = 'cifar-10-batches-py'
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join('./data', 'cifar10')
        super().__init__(root, train, transform, target_transform, download)

    def get_image(self, index):
        return self.data[index]


class CIFAR100(TVCIFAR100):
    base_folder = 'cifar-100-python'
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join('./data', 'cifar100')
        super().__init__(root, train, transform, target_transform, download)

    def get_image(self, index):
        return self.data[index]


class SVHN(TVSVHN):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join('./data', 'svhn')
        # split argument should be one of {‘train’, ‘test’, ‘extra’}
        if isinstance(train, bool):
            split = 'train' if train else 'test'
        else:
            split = train
        super().__init__(root, split, transform, target_transform, download)
        self.classes = list(range(max(self.labels)+1))


class TinyImageNet200(ImageFolder):
    """
    Dataset for TinyImageNet200

    Note: the directory structure slightly varies from original
    To get there, run these two commands:
    - From within tiny-images-200 directory
        for dr in train/*; do
            echo $dr;
            mv $dr/images/* $dr/;
            rmdir $dr/images;
        done
    - From within tiny-images-200/val directory
         while read -r fname label remainder; do
            mkdir -p val2/$label;
            mv images/$fname val2/$label/;
        done < val_annotations.txt

    """

    def __init__(self, train=True, transform=None, target_transform=None, **kwargs):
        root = osp.join('./data', 'tiny-imagenet-200')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://tiny-imagenet.herokuapp.com'
            ))

        # Initialize ImageFolder
        _root = osp.join(root, 'train' if train else 'val/val2')
        super().__init__(root=_root, transform=transform,
                         target_transform=target_transform)
        self.root = root

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

        self._load_meta()

    def _load_meta(self):
        """Replace class names (synsets) with more descriptive labels"""
        # Load mapping
        synset_to_desc = dict()
        fpath = osp.join(self.root, 'words.txt')
        with open(fpath, 'r') as rf:
            for line in rf:
                synset, desc = line.strip().split(maxsplit=1)
                synset_to_desc[synset] = desc

        # Replace
        for i in range(len(self.classes)):
            self.classes[i] = synset_to_desc[self.classes[i]]
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}