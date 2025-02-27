# FedProphet

This is the official implementation of MLSys 2025 paper *FedProphet: Memory-Efficient Federated Adversarial Training via Robust and Consistent Cascade Learning.* 

## Abstract
Federated Adversarial Training (FAT) can supplement robustness against adversarial examples to Federated Learning (FL), promoting a meaningful step toward trustworthy AI. However, FAT requires large models to preserve high accuracy while achieving strong robustness, incurring high memory-swapping latency when training on memory-constrained edge devices. Existing memory-efficient FL methods suffer from poor accuracy and weak robustness due to inconsistent local and global models. In this paper, we propose FedProphet, a novel FAT framework that can achieve memory efficiency, robustness, and consistency simultaneously. FedProphet reduces the memory requirement in local training while guaranteeing adversarial robustness by adversarial cascade learning with strong convexity regularization, and we show that the strong robustness also implies low inconsistency in FedProphet. We also develop a training coordinator on the server of FL, with Adaptive Perturbation Adjustment for utility-robustness balance and Differentiated Module Assignment for objective inconsistency mitigation. FedProphet significantly outperforms other baselines under different experimental settings, maintaining the accuracy and robustness of end-to-end FAT with 80% memory reduction and up to 10.8x speedup in training time.

## Getting Start
### Environment
Please use the following command to build a environment in Anaconda:

```shell
conda env create -f environment.yml
conda activate fedprophet
```

### Datasets
CIFAR-10 will be automatically installed by TorchVision. Use the following command to install [Caltech-256](https://data.caltech.edu/records/nyy15-4j048):

```shell
mkdir ./data
cd ./data
# Caltech256
wget https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar
tar -xf ./256_ObjectCategories.tar
```
## Running Experiments
To run an experiment, use one of the following commands for a specified algorthm on a dataset. 

### CIFAR-10

* jFAT
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=FedBN --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314  --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedDF-AT
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=FedDF --edge_model_archs vgg16_bn vgg13_bn vgg11_bn cnn3 --public_dataset_size=5000 --dist_iters=128 --dist_lr=0.005 --dist_batch_size=64 --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10  --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedET-AT
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=FedET --edge_model_archs vgg16_bn vgg13_bn vgg11_bn cnn3 --public_dataset_size=5000 --dist_iters=128 --dist_lr=0.005 --dist_batch_size=64 --diver_lamb=0.05 --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10  --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* HeteroFL-AT
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=HeteroFL --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10  --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedDrop-AT
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=FedDrop --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10  --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedRolex-AT
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=FedRolex --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10  --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedRBN
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=DBN --pretrained --flalg=FedRBN --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10  --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedProphet
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=FedProphet --mu=1e-5 --adapt_eps  --epochs=3500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10  --adv_ratio=1.0  --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

### Caltech256
* jFAT
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model_arch=resnet34 --norm=BN --pretrained --flalg=FedBN --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=2.24e8 --verbose --seed 1 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedDF-AT
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model_arch=resnet34 --norm=BN --pretrained --flalg=FedDF --edge_model_archs resnet34 resnet18 resnet10 cnn4 --public_dataset_size=2500 --dist_iters=128 --dist_lr=0.001 --dist_batch_size=32 --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=2.24e8 --verbose --seed 1 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedET-AT
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model_arch=resnet34 --norm=BN --pretrained --flalg=FedET --edge_model_archs resnet34 resnet18 resnet10 cnn4 --public_dataset_size=2500 --dist_iters=128 --dist_lr=0.001 --dist_batch_size=32 --diver_lamb=0.05 --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=2.24e8 --verbose --seed 1 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* HeteroFL-AT
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model_arch=resnet34 --norm=BN --pretrained --flalg=HeteroFL --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=2.24e8 --verbose --seed 1 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedDrop-AT
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model_arch=resnet34 --norm=BN --pretrained --flalg=FedDrop --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=2.24e8 --verbose --seed 1 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedRolex-AT
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model_arch=resnet34 --norm=BN --pretrained --flalg=FedRolex --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=2.24e8 --verbose --seed 1 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedRBN
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model_arch=resnet34 --norm=DBN --pretrained --flalg=FedRBN --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=2.24e8 --verbose --seed 1 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedProphet
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model_arch=resnet34 --norm=BN --pretrained --flalg=FedProphet --mu=1e-5 --eps_quantile=0.4 --adapt_eps --epochs=3500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=2.24e8 --verbose --seed 1 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```