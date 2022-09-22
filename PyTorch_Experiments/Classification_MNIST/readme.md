### VTTCG optimizer: Variable three-term conjugate gradient method


### Dependencies
Python 3.8.10
PyTorch 1.10.1
TorchVision 0.11.2


### Training code

(1) train network on MNIST
CUDA_VISIBLE_DEVICES=0 python3 main.py --optim VTTCG --lr 0.001 --total_epoch 100 --batchsize 128

--optim: name of optimizers
--lr: initial learning rate
--epoch: number of epochs to train
--batchsize: the size of the batches


### Running time
On a single GeForce GTX 1070 GPU, training a simple CNN typically takes about 10 minutes.