# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


version =  torch.__version__


# Options
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data_dir', default='/home/zzd/Market/pytorch', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
opt = parser.parse_args()

data_dir = opt.data_dir


# Load Data
transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((288, 144), interpolation=3),
    # transforms.RandomCrop((256, 128)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]

transform_val_list = [
    transforms.Resize(size=(256, 128),interpolation=3), #Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}

train_dir = os.path.join(data_dir, 'train' + ('_all' if opt.train_all else ''))
val_dir = os.path.join(data_dir, 'val')

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(train_dir, data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(val_dir, data_transforms['val'])

dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=opt.batch_size, 
                                            shuffle=(x == 'train'), num_workers=16)
    for x in ['train', 'val']
}

dataset_sizes = {
        x: len(image_datasets[x]) 
    for x in ['train', 'val']
}

class_names = image_datasets['train'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# prepare_dataset
# ------------------
#
# Now, let's write a general function to train a model. Here, we will illustrate:
# -  Scheduling the learning rate
# -  Saving the best model
#
# Parameter ``scheduler`` is an LR scheduler object from `torch.optim.lr_scheduler`.

def prepare_model():
    since = time.time()

    num_epochs = 1
    for epoch in range(num_epochs):
        print('Epoch {} / {}'.format(epoch, num_epochs-1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:

            mean = torch.zeros(3)
            std = torch.zeros(3)

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                B, c, h, w = inputs.shape
                mean += torch.sum(torch.mean(torch.mean(inputs, dim=3), dim=2), dim=0)
                std += torch.sum(torch.std(inputs.view(B,c,h*w), dim=2), dim=0)
                
            print('Mean:', mean / dataset_sizes['train'])
            print('Std:', std / dataset_sizes['train'])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return 


prepare_model()
