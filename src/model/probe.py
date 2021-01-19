import os
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utility import dataset, ToTensor
from resnet18 import Resnet18_MLP

parser = argparse.ArgumentParser(description='our_model')
parser.add_argument('--model', type=str, default='Resnet18_MLP')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--load_workers', type=int, default=16)
parser.add_argument('--resume', type=bool, default=True)
parser.add_argument('--path', type=str, default='../../../RPM/Balanced-RAVEN/70k_dataset')
parser.add_argument('--save', type=str, default='./save_200/')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--meta_alpha', type=float, default=0.0)
parser.add_argument('--meta_beta', type=float, default=0.0)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
torch.cuda.set_device(args.device)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

args.test_figure_configurations = [0,1,2,3,4,5,6]
testds = dataset(args.path, "test", args.img_size, transform=transforms.Compose([ToTensor()]))
testloader = DataLoader(testds, batch_size=args.batch_size, shuffle=False, num_workers=16)
model = Resnet18_MLP(args)
if args.resume:
    model.load_model(args.save, 199)
    print('Loaded model')
if args.cuda:
    model = model.cuda()

def test():
    model.eval()
    accuracy = 0
    acc_all = 0.0
    counter = 0
    for batch_idx, (image, target, meta_target, meta_structure, embedding, indicator) in enumerate(testloader):
        counter += 1
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
            meta_structure = meta_structure.cuda()
            embedding = embedding.cuda()
            indicator = indicator.cuda()
        acc = model.test_(image, target, meta_target, meta_structure, embedding, indicator)
        # print('Test: Epoch:{}, Batch:{}, Acc:{:.4f}.'.format(epoch, batch_idx, acc))  
        acc_all += acc
        if batch_idx % 100 == 0:
                print("interim accuracy, batch", batch_idx, ":")
                print((acc_all / float(counter)))
    if counter > 0:
        print("Total Testing Acc: {:.4f}".format(acc_all / float(counter)))
    return acc_all/float(counter)

def main():
    avg_test_acc = test()

if __name__ == '__main__':
    main()
