# FedProphet

This is a implementation of FedProphet.

## Todos
1. (done) Re-organize the FL framework to make it more scalable.
    1. Dataset Partitioner
    2. FL Clients
    3. Selector
    4. Scheduler
    5. FL Server
2. (done) Implement hardware profiler.
3. (done) Implement FedProphet: Model partitioner, local trainer, training organizer.
4. Implement baselines: (done) FedET, FedDF, FedRolex, HeteroFL with adversarial training.

## Running Commands

### CIFAR-10
* FedAvg
```shell
python3 src/federated_main.py --gpu=2 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=LN --pretrained --flalg=FedAvg --reserved_mem=6.4e7 --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.997 --momentum=0.9 --iid=0 --alpha=1.0  --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --verbose --seed 1 2 3
```

* FedAvgAT
```shell
python3 src/federated_main.py --gpu=3 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=sBN --pretrained --flalg=FedAvg --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.997 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6.4e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedBN
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model_arch=vgg16_bn --pretrained --flalg=FedBN --reserved_mem=6.4e7 --epochs=300 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.01 --lr_decay=0.993 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --verbose --seed 1 2 3
```

* FedBNAT
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=sBN --pretrained --flalg=FedBN --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.997 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6.4e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedProphet

```shell
python3 src/federated_main.py --gpu=3 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=FedProphet --mu=1e-6 --lamb=1e-4 --psi=1.0   --epochs=3500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.997 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6.4e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --eps_quantile=0.3 --adapt_eps --adv_ratio=1.0 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedDFAT
```shell
python3 src/federated_main.py --gpu=1 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=FedDF --edge_model_archs vgg16_bn vgg13_bn vgg11_bn cnn6 --public_dataset_size=5000 --dist_iters=128 --dist_lr=0.005 --dist_batch_size=64 --epochs=1500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.997 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6.4e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedETAT
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=BN --pretrained --flalg=FedET --edge_model_archs vgg16_bn vgg13_bn vgg11_bn cnn6 --public_dataset_size=5000 --dist_iters=128 --dist_lr=0.005 --dist_batch_size=64 --diver_lamb=0.05 --epochs=1500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.997 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6.4e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* HeteroFL
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=sBN --pretrained --flalg=HeteroFL --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.997 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6.4e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=200 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FederatedDropout
```shell
python3 src/federated_main.py --gpu=2 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=sBN --pretrained --flalg=FedDrop --epochs=1500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.997 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6.4e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

* FedRolex
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model_arch=vgg16_bn --norm=sBN --pretrained --flalg=FedRolex --epochs=1500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.005 --lr_decay=0.997 --momentum=0.9 --iid=0 --shards_per_client=2 --skew=0.2 --flsys_profile_info=./src/hardware/flsys_profile_info_low --device_random_seed=717 --sys_scaling_factor=0.0 --reserved_mem=6.4e7 --verbose --seed 1 2 3 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=20
```

### CIFAR-100
1. FedBN
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR100 --model_arch=vgg16_bn --pretrained --flalg=FedBN --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.01 --lr_decay=0.1 --lr_schedule 300 400 --momentum=0.9 --iid=0 --alpha=10.0  --flsys_profile_info=./src/hardware/flsys_profile_info --device_random_seed=717 --sys_scaling_factor=0.0 --verbose --seed 1 2 3
```

### Caltech256
1. FedBN
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model_arch=resnet50 --pretrained --flalg=FedBN --reserved_mem=5.12e8 --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.005 --lr_decay=0.997 --momentum=0.9 --iid=0 --shards_per_client=50 --skew=0.2  --flsys_profile_info=./src/hardware/flsys_profile_info_high --device_random_seed=717 --sys_scaling_factor=0.0 --verbose --seed 1 2 3
```