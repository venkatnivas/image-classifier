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


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    # resize the image
    image = image.resize((256, 256))

    # crop the center
    width, height = image.size

    image = image.crop((16, 16, 240, 240))

    # Change color channels for the image and normalize using transforms
    np_image = np.array(image)
    np_image = np_image / 255

    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])

    np_image = (np_image - mean) / std_dev
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def predict(image_path, model, top_k, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file
    image = process_image(Image.open(image_path))

    image_tensor = torch.from_numpy(image)
    image_tensor.unsqueeze_(0)

    model.eval()

    if device == 'gpu':
        model.to('cuda')
        output = torch.exp(model.forward(image_tensor.to('cuda').float()))
    else:
        output = torch.exp(model.forward(image_tensor.float()))

    results, index = output.topk(top_k)

    prob = results.cpu().data.numpy()
    idx = index.cpu().data.numpy()

    return [p for p in prob[0]], [model.idx_to_class[x] for x in idx[0]]


def main():

    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument(
        'image_path', help='Path of the image to predict')
    parser.add_argument(
        'checkpoint_path', help='Path to load the checkpoint')
    parser.add_argument('--top_k', default=1, type=int,
                        help="Returns the top K classes")
    parser.add_argument('--category_names', default='cat_to_name.json',
                        help='File path for mapping category to real names')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Flag to enable gpu for training models')

    args = parser.parse_args()
    image_path = args.image_path
    checkpoint_path = args.checkpoint_path
    top_k = args.top_k
    category_names = args.category_names

    if args.gpu:
        device = 'gpu'
    else:
        device = 'cpu'

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load checkpoint
    model = load_checkpoint(checkpoint_path)

    # invert key value to map indices to classes
    idx_to_class = dict([[v, k] for k, v in model.class_to_idx.items()])
    model.idx_to_class = idx_to_class

    prob, classes = predict(image_path, model, top_k, device)

    classes = [cat_to_name[x] for x in classes]
    print('\nRESULT:')
    for i in range(top_k):
        print('{}:{:.2%}'.format(classes[i], prob[i]))


if __name__ == "__main__":
    main()
