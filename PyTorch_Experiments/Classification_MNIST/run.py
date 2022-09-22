import os

print('==================VTTCG MNIST==================')
cmd = 'CUDA_VISIBLE_DEVICES=0 python3 main.py --optim VTTCG --lr 0.001 --epoch 100 --batchsize 128'
os.system(cmd)
