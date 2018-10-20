# Build a classifier network
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os.path

# TODO: Build and train your network


class PreTrainedNetwork:

    def __init__(self, model="densenet161", learning_rate=0.001, epochs=5):

        # Load pre-trained network
        # TODO: Change for differen models
        self.model = models.densenet161(pretrained=True)

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # TODO change it for different models

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2208, 512)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(512, 256)),
            ('relu', nn.ReLU()),
            ('fc3', nn.Linear(256, 128)),
            ('relu', nn.ReLU()),
            ('fc4', nn.Linear(128, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        self.model.classifier = classifier

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(
            self.model.classifier.parameters(), lr=self.learning_rate)

    # train the network
    def train(self, trainloader, testloader, device='cpu', print_every=40):

        print_every = print_every
        steps = 0

        # Convert model to device only if cuda or cpu
        if device.lower() == 'gpu':
            self.model.to('cuda')

        for e in range(self.epochs):
            self.model.train()
            running_loss = 0
            for ii, (inputs, labels) in enumerate(trainloader):

                steps += 1

                if device.lower() == 'gpu':
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')

                self.optimizer.zero_grad()

                # Forward and backward passes
                outputs = self.model.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:

                    self.model.eval()

                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        test_loss, accuracy = self._validation(
                            testloader, device)

                    print("Epoch: {}/{}.. ".format(e + 1, self.epochs),
                          "Training Loss: {:.3f}.. ".format(
                              running_loss / print_every),
                          "Test Loss: {:.3f}.. ".format(
                              test_loss / len(testloader)),
                          "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))

                    running_loss = 0

                    # Make sure training is back on
                    self.model.train()

    # Validate the network
    def _validation(self, testloader, device):
        test_loss = 0
        accuracy = 0
        for ii, (images, labels) in enumerate(testloader):

            if device.lower() == 'gpu':
                images, labels = images.to('cuda'), labels.to('cuda')

            output = self.model.forward(images)
            test_loss += self.criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return test_loss, accuracy

    # Save checkpoint

    def save_checkpoint(self, directory, data_transforms, image_datasets):
        # TODO: Save the checkpoint
        checkpoint = {'input_size': 2208,
                      'output_size': 102,
                      'epochs': self.epochs,
                      'learning_rate': self.learning_rate,
                      'batch_size': 64,
                      'data_transforms': data_transforms,
                      'model': self.model,
                      'classifier': self.model.classifier,
                      'optimizer': self.optimizer.state_dict(),
                      'state_dict': self.model.state_dict(),
                      'class_to_idx': image_datasets.class_to_idx
                      }

        torch.save(checkpoint, os.path.join(directory, 'checkpoint.pth'))
