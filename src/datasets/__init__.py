from torchvision import transforms

from datasets.cifarlike import CIFAR10, CIFAR100, SVHN, TinyImageNet200
from datasets.imagenetlike import ImageNet1k,ImageNette,CUBS200,Caltech101,Caltech256
from datasets.mnistlike import MNIST, KMNIST, FashionMNIST
from datasets.language import shakespeare, sent140


# Create a mapping of dataset -> dataset_type
# This is helpful to determine which (a) family of model needs to be loaded e.g., imagenet and
# (b) input transform to apply
dataset_to_datafamily = {
    # MNIST
    'MNIST': 'mnist',
    'KMNIST': 'mnist',
    'FashionMNIST': 'mnist',

    # Cifar
    'CIFAR10': 'cifar',
    'CIFAR100': 'cifar',
    'SVHN': 'cifar',
    'TinyImageNet200': 'cifar',
    

    # Imagenet
    'CUBS200': 'imagenet',
    'Caltech101': 'imagenet',
    'Caltech256': 'imagenet',
    'ImageNet1k': 'imagenet',
    'ImageNette': 'imagenet',

    # NLP datasets
    'shakespeare': 'nlp',
    'sent140': 'nlp'
}

datafamily_to_normalize = {
    'mnist': transforms.Normalize(mean=(0.1307,), 
                                  std=(0.3081,)),
    'cifar': transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                  std=(0.2023, 0.1994, 0.2010)),
    'imagenet': transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
}

# Transforms
datafamily_to_transforms = {
    'mnist': {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'avd_train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'adv_test': transforms.Compose([
            transforms.ToTensor()
        ])
    },

    'cifar': {
        'train': transforms.Compose([
            transforms.Resize([32,32]),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010)),
        ]),
        'adv_train': transforms.Compose([
            transforms.Resize([32,32]),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize([32,32]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                 std=(0.2023, 0.1994, 0.2010)),
        ]),
        'adv_test': transforms.Compose([
            transforms.Resize([32,32]),
            transforms.ToTensor()
        ])
    },

    'imagenet': {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'adv_train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'adv_test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    },

    
}

