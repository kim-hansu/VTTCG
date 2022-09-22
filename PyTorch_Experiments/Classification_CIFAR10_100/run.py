import os

print('==================VTTCG CIFAR-100==================')
cmd = 'CUDA_VISIBLE_DEVICES=0 python3 main.py --optim VTTCG --lr 0.001 --total_epoch 100 --batchsize 256 --model wideresnet --dataset cifar100'
os.system(cmd)

print('==================VTTCG CIFAR-10==================')
cmd = 'CUDA_VISIBLE_DEVICES=0 python3 main.py --optim VTTCG --lr 0.001 --total_epoch 100 --batchsize 256 --model resnet --dataset cifar10'
os.system(cmd)