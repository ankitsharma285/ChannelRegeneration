from __future__ import print_function
import os, time
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from TinyImageNet import TinyImageNet
import torchvision.models as models
# import models
# # from filter import *
# # from scipy.ndimage import filters
# from compute_flops import print_model_param_flops
# from torchsummary import summary

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=90, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--arch', default='resnet18', type=str,
                    help='architecture to use')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--rg_val', type=float, default=0.5, metavar='LR',
                    help='RG Val (default: 0.5)')
parser.add_argument('--rg_epoch', type=float, default=10, metavar='LR',
                    help='RG Epoch (default: 10)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=900, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')

# parser.add_argument("--local_rank", type=int, default=0)
# parser.add_argument("--port", type=str, default="15000")
root = '/home/sharma/tiny-imagenet-200'

args = parser.parse_args()

torch.manual_seed(111)
torch.cuda.manual_seed(111)

def rejuvenate():
    # Get the global bn_threshold for the pruned model
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * args.rg_val)
    avg_thre = y[thre_index]

    # Procedure to reset bn_scale term of channels
    for k,m in enumerate(model.modules()):
        if isinstance(m,nn.BatchNorm2d):
            reset_tensor = torch.ones_like(m.weight.data) * avg_thre
            m.weight.data = torch.where(m.weight.data.abs() < avg_thre.cuda(),reset_tensor,m.weight.data)


# torch.distributed.init_process_group(backend="nccl")
# torch.distributed.init_process_group(backend="gloo")
# os.environ['MASTER_PORT'] = args.port

kwargs = {'num_workers': 1, 'pin_memory': True}
normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))

augmentation = transforms.RandomApply([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(64)], p=.8)

training_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    augmentation,
    transforms.ToTensor(),
    normalize])

valid_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    normalize])

in_memory = False
training_set = TinyImageNet(root, 'train', transform=training_transform, in_memory=in_memory)
valid_set = TinyImageNet(root, 'val', transform=valid_transform, in_memory=in_memory)
train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(valid_set, batch_size=64, num_workers=6)

if args.arch == 'mobilenet_v2':
    model = models.mobilenet_v2(pretrained=False,num_classes=200)
elif args.arch == 'resnet18':
    model = models.resnet18(pretrained=False,num_classes=200)
elif args.arch == 'shufflenet':
    model = models.shufflenet_v2_x1_0(pretrained=False,num_classes=200)
    
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        train_acc += prec1.item()
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    test_acc = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()

    test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, test_acc, len(test_loader), test_acc / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2)

best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    prec1 = test()
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    if args.rg_epoch != 0:
        if(epoch % args.rg_epoch == 0):
            rejuvenate()


print("Arch: {} Best accuracy: {:.3f} RG epoch: {} RG Val: {:.2f}".format(args.arch,best_prec1,args.rg_epoch,args.rg_val))
