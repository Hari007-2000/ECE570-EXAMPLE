#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
import shutil
import datetime
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
from utils import *
from model import *

# Argument parser
parser = argparse.ArgumentParser(description='variance scaled weight perturbation')
parser.add_argument("-s", "--seed", type=int, default=42, help="The random seed")
parser.add_argument("-l", "--learning_rate", type=float, default=0.01, help="The learning rate")
parser.add_argument("-e", "--epoch", type=int, default=200, help="The total epoch")
parser.add_argument('--ratio', default=0.8, type=float, help='hyperparameter lambda for Adversarial Sampling')
parser.add_argument('--dataset', default='CIFAR-10', type=str, help='dataset')
parser.add_argument('--model', default='resnet_vi', type=str, help='network structure')
parser.add_argument('--samplings', default=1, type=int, help='sampling times')
parser.add_argument('--pretrain', default=False, type=bool, help='use pre training and Bayes finetune')
parser.add_argument('--notvi', help='not use BNN', dest="notvi", action='store_true')
args = parser.parse_args()

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Setup data paths and device
datapath = os.path.expanduser("~/data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # Start time logging
    start_time = datetime.datetime.now()
    print("Start time:", start_time.isoformat())
    
    # Seed setup
    set_seed(args.seed)
    
    # Log some argument information
    print("Using configuration: ")
    print(f"Random seed: {args.seed}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Total epochs: {args.epoch}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Samplings: {args.samplings}")
    
    # Set up result folder paths
    signature = "ratio"
    rootpath = f"results/{signature}_seed{args.seed}_l{args.learning_rate}_epoch{args.epoch}_ratio{args.ratio}_dataset{args.dataset}_model{args.model}_samplings{args.samplings}"
    if args.pretrain:
        rootpath += "_pretrain"
    if args.notvi:
        rootpath += "_notvi"
    os.makedirs(rootpath, exist_ok=True)
    
    # Save backup files in result folder
    shutil.copyfile("main.py", os.path.join(rootpath, "main_bak.py"))
    shutil.copyfile("utils.py", os.path.join(rootpath, "utils_bak.py"))
    shutil.copyfile("model.py", os.path.join(rootpath, "model_bak.py"))

    # Define dataset transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    # Load datasets
    if args.dataset == 'CIFAR-10':
        trainset = datasets.CIFAR10(root=datapath, train=True, download=True, transform=transform_train)
        valset = datasets.CIFAR10(root=datapath, train=False, download=True, transform=transform_test)
        num_classes = 10
    elif args.dataset == 'CIFAR-100':
        trainset = datasets.CIFAR100(root=datapath, train=True, download=True, transform=transform_train)
        valset = datasets.CIFAR100(root=datapath, train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise NotImplementedError('Invalid dataset')

    # Data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, pin_memory=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, pin_memory=True, num_workers=0)

    # Model selection
    if 'resnet' in args.model:
        net = ResNetWrapper(N=len(trainset), num_classes=num_classes, net=args.model, vi=not args.notvi)
    elif 'vgg' in args.model:
        net = VGG16VIWrapper(N=len(trainset), num_classes=num_classes, vi=not args.notvi)
    else:
        raise NotImplementedError('Invalid model')
    
    net.model.to(device)

    # Training variables
    train_losses = []
    train_precs = []
    test_losses = []
    test_precs = []
    lr = args.learning_rate

    # Pretraining
    if args.pretrain:
        pretrain_path = f"pretrain/{args.dataset}-{args.model}/{args.model}.pt"
        net_dict = net.model.state_dict()
        pretrain_dict = torch.load(pretrain_path, map_location=device)['state_dict'].items()
        net_dict.update({k.replace('weight', 'mu_weight').replace('bias', 'mu_bias'): v for k, v in pretrain_dict})
        net.model.load_state_dict(net_dict)
        print("Loaded pretrained model from", pretrain_path)
        net.validate(valloader, sample=False)

    # Training loop
    for epoch in range(args.epoch):
        if epoch in [80, 140, 180]:
            lr /= 10  # Adjust learning rate
            
        # Train the model
        loss, prec = net.fit(trainloader, lr=lr, epoch=epoch, optimizer="adam", ratio=args.ratio, samplings=args.samplings)
        train_losses.append(loss)
        train_precs.append(prec)

        # Validate the model
        loss, prec = net.validate(valloader)
        test_losses.append(loss)
        test_precs.append(prec)

    # Results logging
    print("Best validation accuracy:", max(test_precs))
    net.save(os.path.join(rootpath, f"{args.model}.pt"))
    end_time = datetime.datetime.now()
    print("End time:", end_time.isoformat())
    print("Total time:", end_time - start_time)



