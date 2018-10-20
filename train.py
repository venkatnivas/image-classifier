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


def getModel(arch):
    # Dictionary mapping for the supported pretrained models
    input_size_dict = {
        'densenet121': (models.densenet121(pretrained=True), 1024),
        'densenet161': (models.densenet161(pretrained=True), 2208),
        'densenet169': (models.densenet169(pretrained=True), 1664),
        'densenet201': (models.densenet201(pretrained=True), 1920),
        'vgg11': (models.vgg11(pretrained=True), 25088),
        'vgg13': (models.vgg13(pretrained=True), 25088),
        'vgg16': (models.vgg16(pretrained=True), 25088),
        'vgg19': (models.vgg19(pretrained=True), 25088),
    }

    return input_size_dict[arch]


def main():
    # argparse
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument(
        'data_directory', help='data directory to train the network')
    parser.add_argument('--save_dir', default='save_directory',
                        help='directory to save checkpoint')
    parser.add_argument('--epochs', default=3, type=int,
                        help='Number of epochs to train the model')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Learning rate for the model to train on the training set')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Flag to enable gpu for training models')
    parser.add_argument('--hidden_units', nargs='+', type=int,
                        default=[512, 256, 128], help='The hidden units for the network')
    parser.add_argument('--arch', default='densenet161', type=str,
                        help='Pretrained Model to use for the network')
    args = parser.parse_args()
    data_dir = args.data_directory
    save_dir = args.save_dir
    learning_rate = args.learning_rate
    epochs = args.epochs
    hidden_units = args.hidden_units
    arch = args.arch

    # Get the model and input_size
    try:
        model, input_size = getModel(arch)
    except:
        print('For now, supporting densenet and vgg only')
        sys.exit()

    # Idx to name mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Output size of the network
    output_size = len(cat_to_name)

    # Device used for training
    if args.gpu:
        device = 'gpu'
    else:
        device = 'cpu'

    # create directory to save checkpoint
    cwd = os.getcwd()
    checkpoint_path = os.path.join(cwd, save_dir)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # transform images and get train, test and validation datasets
    data_transforms, image_datasets, dataloaders = load_data(data_dir)

    # Use a pretrained network for the model
    my_network = PreTrainedNetwork(
        model, input_size, output_size, hidden_units, learning_rate, epochs)

    # train the network
    my_network.train(dataloaders['train_loaders'],
                     dataloaders['validation_loaders'], device)

    # save checkpoint
    my_network.save_checkpoint(
        checkpoint_path, data_transforms, image_datasets['train_dataset'])


if __name__ == "__main__":
    main()
