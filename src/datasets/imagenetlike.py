#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import os.path as osp
import numpy as np
from collections import defaultdict as dd

from torchvision.datasets import ImageFolder
from torchvision.datasets import ImageNet as TVImageNet


class ImageNet1k(TVImageNet):
    root = "/home/public/ImageNet"
    def __init__(self, train=True, transform=None, target_transform=None,**kwargs):
        split = 'train' if train else 'val'
        # root = osp.join(cfg.DATASET_ROOT, 'ILSVRC2012')
        if not osp.exists(self.root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                self.root, 'http://image-net.org/download-images'
            ))
        super().__init__(self.root, split, transform=transform, target_transform=target_transform,download=False)

        # print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
        #                                                         len(self.samples)))


class ImageNette(ImageFolder):

    def __init__(self, train=True, transform=None, target_transform=None,**kwargs):
        root = osp.join("./data", 'imagenette2-320')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
            ))

        # Initialize ImageFolder
        super().__init__(root=osp.join(root, 'train' if train else 'val'), transform=transform,
                         target_transform=target_transform)
        self.root = root

        

class CUBS200(ImageFolder):
    def __init__(self, train=True, transform=None, target_transform=None,**kwargs):
        root = osp.join('./data', 'CUB_200_2011')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://data.caltech.edu/records/65de6-vp158'
            ))

        # Initialize ImageFolder
        super().__init__(root=osp.join(root, 'images'), transform=transform,
                         target_transform=target_transform)
        self.root = root

        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        # print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
        #                                                         len(self.samples)))

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # ----------------- Create mapping: filename -> 'train' / 'test'
        # There are two files: a) images.txt containing: <imageid> <filepath>
        #            b) train_test_split.txt containing: <imageid> <0/1>

        imageid_to_filename = dict()
        with open(osp.join(self.root, 'images.txt')) as f:
            for line in f:
                imageid, filepath = line.strip().split()
                _, filename = osp.split(filepath)
                imageid_to_filename[imageid] = filename
        filename_to_imageid = {v: k for k, v in imageid_to_filename.items()}

        imageid_to_partition = dict()
        with open(osp.join(self.root, 'train_test_split.txt')) as f:
            for line in f:
                imageid, split = line.strip().split()
                imageid_to_partition[imageid] = 'train' if int(split) else 'test'

        # Loop through each sample and group based on partition
        for idx, (filepath, _) in enumerate(self.samples):
            _, filename = osp.split(filepath)
            imageid = filename_to_imageid[filename]
            partition_to_idxs[imageid_to_partition[imageid]].append(idx)

        return partition_to_idxs

class Caltech101(ImageFolder):

    def __init__(self, train=True, transform=None, target_transform=None,**kwargs):
        root = osp.join('./data', '101_ObjectCategories')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://data.caltech.edu/records/mzrjq-6wc02'
            ))

        # Initialize ImageFolder
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        self._cleanup()
        self.ntest = 15  # Reserve these many examples per class for evaluation
        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        # print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
        #                                                         len(self.samples)))

    def _cleanup(self):
        # Remove examples belonging to class "clutter"
        background_idx = self.class_to_idx['BACKGROUND_Google']
        self.samples = [s for s in self.samples if s[1] != background_idx]
        del self.class_to_idx['BACKGROUND_Google']
        self.classes.remove('BACKGROUND_Google')

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        

        # ----------------- Create mapping: classidx -> idx
        classidx_to_idxs = dd(list)
        for idx, s in enumerate(self.samples):
            classidx = s[1]
            classidx_to_idxs[classidx].append(idx)

        # Shuffle classidx_to_idx
        for classidx, idxs in classidx_to_idxs.items():
            np.random.shuffle(idxs)

        for classidx, idxs in classidx_to_idxs.items():
            partition_to_idxs['test'] += idxs[:self.ntest]     # A constant no. kept aside for evaluation
            partition_to_idxs['train'] += idxs[self.ntest:]    # Train on remaining


        return partition_to_idxs


class Caltech256(ImageFolder):

    def __init__(self, train=True, transform=None, target_transform=None,**kwargs):
        root = osp.join('./data', '256_ObjectCategories')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://data.caltech.edu/records/nyy15-4j048'
            ))

        # Initialize ImageFolder
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        self._cleanup()
        self.ntest = 25  # Reserve these many examples per class for evaluation
        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        # print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
        #                                                         len(self.samples)))

    def _cleanup(self):
        # Remove examples belonging to class "clutter"
        clutter_idx = self.class_to_idx['257.clutter']
        self.samples = [s for s in self.samples if s[1] != clutter_idx]
        del self.class_to_idx['257.clutter']
        self.classes = self.classes[:-1]

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        

        # ----------------- Create mapping: classidx -> idx
        classidx_to_idxs = dd(list)
        for idx, s in enumerate(self.samples):
            classidx = s[1]
            classidx_to_idxs[classidx].append(idx)

        # Shuffle classidx_to_idx
        for classidx, idxs in classidx_to_idxs.items():
            np.random.shuffle(idxs)

        for classidx, idxs in classidx_to_idxs.items():
            partition_to_idxs['test'] += idxs[:self.ntest]     # A constant no. kept aside for evaluation
            partition_to_idxs['train'] += idxs[self.ntest:]    # Train on remaining


        return partition_to_idxs

