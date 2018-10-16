#import packages

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

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

#Load directory
def load_dir(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    return train_dir,valid_dir,test_dir

def main():
    #argparse
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument('data_directory', help='directory containing the images to train the network')
    args = parser.parse_args()
    data_dir = args.data_directory
    train_dir,valid_dir,test_dir = load_dir(data_dir)
    print(train_dir,valid_dir,test_dir)

if __name__ == "__main__":
    main()
