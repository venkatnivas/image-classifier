#import packages

import numpy as np
import pandas
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import json

from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

import argparse
import sys
import os
from network import PreTrainedNetwork

# Load directory


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train_transforms': transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(
                                                    224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])]),
        'test_transforms': transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])]),
        'validation_transforms': transforms.Compose([transforms.Resize(256),
                                                     transforms.CenterCrop(
                                                         224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train_dataset': datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
        'test_dataset': datasets.ImageFolder(test_dir, transform=data_transforms['test_transforms']),
        'validation_dataset': datasets.ImageFolder(valid_dir, transform=data_transforms['validation_transforms'])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train_loaders': torch.utils.data.DataLoader(image_datasets['train_dataset'], batch_size=64, shuffle=True),
        'test_loaders': torch.utils.data.DataLoader(image_datasets['test_dataset'], batch_size=64, shuffle=True),
        'validation_loaders': torch.utils.data.DataLoader(image_datasets['validation_dataset'], batch_size=64, shuffle=True)
    }

    return data_transforms, image_datasets, dataloaders


def main():
    # argparse
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument(
        'data_directory', help='data directory to train the network')
    parser.add_argument('--save_dir', default='save_directory',
                        help='directory to save checkpoint')
    parser.add_argument('--epochs', default=5,
                        help='Number of epochs to train the model')
    parser.add_argument('--learning_rate', default=0.001,
                        help='Learning rate for the model to train on the training set')
    args = parser.parse_args()
    data_dir = args.data_directory
    save_dir = args.save_dir
    learning_rate = args.learning_rate
    epochs = args.epochs

    # create directory to save checkpoint
    cwd = os.getcwd()
    checkpoint_path = os.path.join(cwd, save_dir)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    data_transforms, image_datasets, dataloaders = load_data(data_dir)

    my_network = PreTrainedNetwork("densenet161", learning_rate, epochs)
    my_network.train(dataloaders['train_loaders'],
                     dataloaders['validation_loaders'], 'gpu')

    # save checkpoint
    my_network.save_checkpoint(
        checkpoint_path, data_transforms, image_datasets['train_dataset'])


if __name__ == "__main__":
    main()
