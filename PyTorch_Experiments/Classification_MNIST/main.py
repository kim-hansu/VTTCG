from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from optimizers import *
from torch.optim import Adam, SGD
import os


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(3 * 3 * 128, 625)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        output = F.log_softmax(out, dim=1)
        return output


def create_optimizer(args, model_params):
    args.optim = args.optim.lower()
    if args.optim == 'SGD':
        return SGD(model_params, lr=args.lr, momentum=args.momentum)
    elif args.optim == 'Adam':
        return Adam(model_params, lr=args.lr)
    elif args.optim == 'AdaBelief':
        return AdaBelief(model_params, lr=args.lr)
    elif args.optim == 'VTTCG':
        return VTTCG(model_params, lr=args.lr)
    else:
        print('Optimizer not found')


def get_ckpt_name(comp='COMP', dataset='MNIST', batchsize=128, optimizer='VTTCG', epoch=100):
    return '{}-{}-{}--batch-{}-epoch-{}'.format(comp, dataset, optimizer, batchsize, epoch)


def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(path)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--optim', default='VTTCG', type=str, help='optimizer')
    parser.add_argument('--batchsize', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batchsize', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epoch', type=int, default=14, metavar='N',
                        help='number of epoch to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    return parser


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    train_loss += loss.item()
    print('\nEpoch: {}, Trian loss: {:.10f}'.format(epoch, train_loss))

    return train_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


def main():
    # Training settings
    parser = get_parser()
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Save the curve
    ckpt_name = get_ckpt_name(batchsize=args.batchsize, optimizer=args.optim, epoch=args.epoch)
    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        start_epoch = ckpt['epoch']

        curve = os.path.join('curve', ckpt_name)
        curve = torch.load(curve)
        train_losses = curve['train_loss']
        test_losses = curve['test_loss']
        test_accuracies = curve['test_acc']
    else:
        ckpt = None
        start_epoch = -1
        train_losses = []
        test_losses = []
        test_accuracies = []

    train_kwargs = {'batchsize': args.batchsize}
    test_kwargs = {'batchsize': args.test_batchsize}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = CNN().to(device)
    optimizer = create_optimizer(args, model.parameters())
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epoch + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(model, device, test_loader)

        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        if not os.path.isdir('curve'):
            os.mkdir('curve')
        torch.save({'train_loss': train_losses, 'test_acc': test_accuracies},
                   os.path.join('curve', ckpt_name))
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()