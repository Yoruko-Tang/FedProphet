# FedProphet

This is a implementation of FedProphet.

## Todos
1. (done) Re-organize the FL framework to make it more scalable.
    1. Dataset Partitioner
    2. FL Clients
    3. Selector
    4. Scheduler
    5. FL Server
2. (to be done) Implement hardware profiler.
3. Implement FedProphet: Model partitioner, local trainer, training organizer.
4. Implement baselines: FedET, FedRolex with adversarial training.

## Running Commands

### CIFAR-10

1. FedBN
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model=vgg16_bn --pretrained --flalg=FedBNAT --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.01 --lr_decay=0.1 --lr_schedule 300 400 --momentum=0.9 --iid=0 --alpha=1.0  --flsys_profile_info=./src/hardware/flsys_profile_info --device_random_seed=717 --sys_scaling_factor=0.0 --verbose --seed 1 2 3
```

2. FedBNAT
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model=vgg16_bn --pretrained --flalg=FedBNAT --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.01 --lr_decay=0.1 --lr_schedule 300 400 --momentum=0.9 --iid=0 --alpha=1.0  --flsys_profile_info=./src/hardware/flsys_profile_info --device_random_seed=717 --sys_scaling_factor=0.0 --verbose --seed 1 2 3 --adv_train --adv_warmup=100 --adv_method=PGD --adv_epsilon=0.0314 --adv_alpha=0.0078 --adv_T=10 --adv_test --advt_method=PGD --advt_epsilon=0.0314 --advt_alpha=0.0078 --advt_T=10 --test_every=5
```

### CIFAR-100
1. FedBN
```shell
python3 src/federated_main.py --gpu=0 --dataset=CIFAR100 --model=vgg16_bn --pretrained --flalg=FedBN --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=64 --optimizer=sgd --lr=0.01 --lr_decay=0.1 --lr_schedule 300 400 --momentum=0.9 --iid=0 --alpha=10.0  --flsys_profile_info=./src/hardware/flsys_profile_info --device_random_seed=717 --sys_scaling_factor=0.0 --verbose --seed 1 2 3
```

### Caltech256
1. FedBN
```shell
python3 src/federated_main.py --gpu=0 --dataset=Caltech256 --model=resnet50 --pretrained --flalg=FedBN --epochs=500 --num_user=100 --frac=0.1 --strategy=rand --local_ep=30 --local_bs=32 --optimizer=sgd --lr=0.01 --lr_decay=0.1 --lr_schedule 300 400 --momentum=0.9 --iid=0 --alpha=25.6  --flsys_profile_info=./src/hardware/flsys_profile_info --device_random_seed=717 --sys_scaling_factor=0.0 --verbose --seed 1 2 3
```