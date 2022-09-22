import os

print('==================VTTCG GAN==================')
os.system("CUDA_VISIBLE_DEVICES=0 python3 main.py --lr 0.001 --total_epoch 100 --batchsize 64")
