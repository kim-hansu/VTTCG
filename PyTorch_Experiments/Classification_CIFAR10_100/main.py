from __future__ import print_function

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import time
from models import *
from torch.optim import Adam, SGD, NAdam
from optimizers import *
from tensorboardX import SummaryWriter

writer = SummaryWriter('./log')
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
    parser.add_argument('--total_epoch', default=100, type=int, help='Total number of training epochs')
    parser.add_argument('--model', default='resnet', type=str, help='model',
                        choices=['resnet', 'densenet', 'vgg', 'wideresnet'])
    parser.add_argument('--optim', default='VTTCG', type=str, help='optimizer')
    parser.add_argument('--run', default=0, type=int, help='number of runs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='learning rate')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps for var adam')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta', default=0.999, type=float, help='VTTCG coefficients beta')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta2')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batchsize', type=int, default=256, help='batch size')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--reset', action = 'store_true',
                        help='whether reset optimizer at learning rate decay')
    return parser


def build_dataset(args):
    print('==> Preparing data..')
    parser = get_parser()
    args = parser.parse_args()
    if (args.dataset == 'cifar10'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)

    elif(args.dataset=='cifar100'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_ckpt_name(comp='COMP', model='resnet', dataset='cifar10', batchsize=128,optimizer='sgd', lr=0.1, total_epoch=100, run = 0):
    return '{}-{}-{}-lr-{}-batch-{}-{}-total-{}-run-{}'.format(comp, dataset, optimizer, lr, batchsize, model, total_epoch,run)


def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(path)


def build_model(args, device, ckpt=None):
    print('==> Building model..')
    net = {
        'resnet': ResNet34,
        'densenet': DenseNet121,
        'vgg':vgg11,
        'wideresnet':Wide_ResNet # for CIFAR 100
    }[args.model]()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net


def create_optimizer(args, model_params):
    args.optim = args.optim.lower()
    if args.optim == 'SGD':
        return SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        return Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'AMSGrad':
        return Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps, amsgrad=True)
    elif args.optim == 'AdaBelief':
        return AdaBelief(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'VTTCG':
        return VTTCG(model_params, args.lr, beta=args.beta, weight_decay=args.weight_decay)
    else:
        print('Optimizer not found')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(net, epoch, device, data_loader, optimizer, criterion, args):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if epoch == 0 or epoch == args.total_epoch - 1:
            learning_rate = get_lr(optimizer)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print('train loss: %.3f' % train_loss)

    return train_loss


def test(net, device, data_loader, criterion):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print('test acc: %.3f' % accuracy)
    return accuracy

def main():
    parser = get_parser()
    args = parser.parse_args()

    train_loader, test_loader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_name = get_ckpt_name(model=args.model,dataset=args.dataset, batchsize=args.batchsize,optimizer=args.optim, lr=args.lr,
                              run = args.run, total_epoch=args.total_epoch)
    print('ckpt_name')
    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        start_epoch = ckpt['epoch']

        curve = os.path.join('curve', ckpt_name)
        curve = torch.load(curve)
        train_losses = curve['train_loss']
        test_accuracies = curve['test_acc']
    else:
        ckpt = None
        start_epoch = -1
        train_losses = []
        test_accuracies = []

    net = build_model(args, device, ckpt=ckpt)
    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(args, net.parameters())

    start_time = time.time()

    for epoch in range(start_epoch + 1, args.total_epoch):
        start = time.time()
        train_loss = train(net, epoch, device, train_loader, optimizer, criterion, args)
        test_acc = test(net, device, test_loader, criterion)
        end = time.time()
        print('Time {}'.format(end-start))

        train_losses.append(train_loss)
        test_accuracies.append(test_acc)


        if not os.path.isdir('curve'):
            os.mkdir('curve')
        torch.save({'train_loss': train_losses, 'test_acc': test_accuracies}, os.path.join('curve', ckpt_name))

    end_time = time.time()
    print('End Time: {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()
