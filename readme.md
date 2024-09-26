# FedProphet

This is a implementation of FedProphet.


## Running Commands

### CIFAR-10

* FedAvgAT
```shell
python3 src/federated_main.py --gpu=3 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=sBN --pretrained --flalg=FedAvg --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```


* FedBNAT

largest
```shell
python3 src/federated_main.py --gpu=3 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=FedBN --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314  --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

smallest
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model_arch=cnn3 --norm=BN --pretrained --flalg=FedBN --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10  --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedProphet

```shell
python3 src/federated_main.py --gpu=3 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=FedProphet --mu=1e-4 --lamb=1e-4 --psi=1.0 --eps_quantile=0.3 --adapt_eps  --epochs=3500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10  --adv_ratio=1.0  --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedDFAT
```shell
python3 src/federated_main.py --gpu=2 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=FedDF --edge_model_archs vgg16_bn vgg13_bn vgg11_bn cnn3 --public_dataset_size=5000 --dist_iters=128 --dist_lr=0.005 --dist_batch_size=64 --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10  --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedETAT
```shell
python3 src/federated_main.py --gpu=2 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=FedET --edge_model_archs vgg16_bn vgg13_bn vgg11_bn cnn3 --public_dataset_size=5000 --dist_iters=128 --dist_lr=0.005 --dist_batch_size=64 --diver_lamb=0.05 --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10  --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* HeteroFL
```shell
python3 src/federated_main.py --gpu=1 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=HeteroFL --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10  --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FederatedDropout
```shell
python3 src/federated_main.py --gpu=3 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=FedDrop --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10  --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedRolex
```shell
python3 src/federated_main.py --gpu=2 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=FedRolex --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10  --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedRBN
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=DBN --pretrained --flalg=FedRBN --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.994 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=-1.0 --reserved_mem=6e7 --verbose --seed 1 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10  --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```


### Caltech256
* FedBNAT
Large
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model_arch=resnet34 --norm=BN --pretrained --flalg=FedBN --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=2.24e8 --verbose --seed 1 2 3 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

Small
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model_arch=cnn4 --norm=BN --pretrained --flalg=FedBN --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=2.24e8 --verbose --seed 1 2 3 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedProphet
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model_arch=resnet34 --norm=BN --pretrained --flalg=FedProphet --mu=1e-5 --lamb=1e-4 --psi=1.0 --eps_quantile=0.4 --adapt_eps --epochs=3500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=2.24e8 --verbose --seed 1 2 3 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedDF
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model_arch=resnet34 --norm=BN --pretrained --flalg=FedDF --edge_model_archs resnet34 resnet18 resnet10 cnn4 --public_dataset_size=2500 --dist_iters=128 --dist_lr=0.001 --dist_batch_size=32 --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=2.24e8 --verbose --seed 1 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedET
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model_arch=resnet34 --norm=BN --pretrained --flalg=FedET --edge_model_archs resnet34 resnet18 resnet10 cnn4 --public_dataset_size=2500 --dist_iters=128 --dist_lr=0.001 --dist_batch_size=32 --diver_lamb=0.05 --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=2.24e8 --verbose --seed 1 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* HeteroFL
```shell
python3 src/federated_main.py --gpu=1 --dataset=Caltech256 --model_arch=resnet34 --norm=BN --pretrained --flalg=HeteroFL --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=-1.0 --reserved_mem=2.24e8 --verbose --seed 1 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedDrop
```shell
python3 src/federated_main.py --gpu=1 --dataset=Caltech256 --model_arch=resnet34 --norm=BN --pretrained --flalg=FedDrop --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=2.24e8 --verbose --seed 1 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedRolex
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model_arch=resnet34 --norm=BN --pretrained --flalg=FedRolex --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=2.24e8 --verbose --seed 1 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedRBN
```shell
python3 src/federated_main.py --gpu=2 --dataset=Caltech256 --model_arch=resnet34 --norm=DBN --pretrained --flalg=FedRBN --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.001 --momentum=0.9 --lr_decay=0.994 --iid=0 --shards_per_client=46 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_mid --device_random_seed=717 --sys_scaling_factor=-1.0 --reserved_mem=2.24e8 --verbose --seed 1 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```