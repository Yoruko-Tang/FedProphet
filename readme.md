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
python3 src/federated_main.py --gpu=0 --dataset=CIFAR10 --model=vgg16_bn --pretrained --flalg=FedBN --epochs=1000 --num_user=100 --frac=0.1 --strategy=rand --local_ep=50 --local_bs=10 --optimizer=sgd --lr=0.01 --lr_decay=1.0 --momentum=0.5 --iid=0 --unequal=0 --alpha=0.1 --flsys_profile_info=./flsys_profile_info --device_random_seed=717 --sys_scaling_factor=0.0 --verbose --seed 1 2 3
```
