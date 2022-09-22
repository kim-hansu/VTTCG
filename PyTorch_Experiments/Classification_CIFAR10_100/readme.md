### VTTCG optimizer: Variable three-term conjugate gradient method


### Dependencies
Python 3.8.10
PyTorch 1.10.1
TorchVision 0.11.2


### Training code

(1) train network on CIFAR-100
CUDA_VISIBLE_DEVICES=0 python3 main.py --optim VTTCG --lr 0.001 --total_epoch 100 --batchsize 256 --model wideresnet --dataset cifar100

(2) train network on CIFAR-10
CUDA_VISIBLE_DEVICES=0 python3 main.py --optim VTTCG --lr 0.001 --total_epoch 100 --batchsize 256 --model resnet --dataset cifar10

--optim: name of optimizers
--lr: initial learning rate
--total_epoch: number of epochs to train
--batchsize: the size of the batches
--model: training network
--dataset: training dataset


### Running time
On a single GeForce GTX 1070 GPU, training a ResNet typically takes about 3 hours, and training WideResNet usually takes about 5 hours
