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

    def __init__(self, model, input_size, output_size, hidden_units, learning_rate, epochs):

        # Load pre-trained network
        # TODO: Change for differen models
        # self.model = models.densenet161(pretrained=True)
        self.model = model
        self.input_size = input_size
        self.output_size = output_size

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

        #
        _hidden_layers = OrderedDict()

        _hidden_layers.update(
            {'fc1': nn.Linear(self.input_size, hidden_units[0])})

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
        count = 2
        for h1, h2 in layer_sizes:
            _hidden_layers.update({'fc' + str(count): nn.Linear(h1, h2)})
            _hidden_layers.update({'relu' + str(count): nn.ReLU()})
            count += 1
        _hidden_layers.update(
            {'fc' + str(count): nn.Linear(hidden_units[-1], self.output_size)})

        _hidden_layers.update({'output': nn.LogSoftmax(dim=1)})

        self.model.classifier = nn.Sequential(_hidden_layers)

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(
            self.model.classifier.parameters(), lr=self.learning_rate)

    def train(self, trainloader, testloader, device='cpu', print_every=40):
        # train the network
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

    def _validation(self, testloader, device):
        # Validate network
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

    def save_checkpoint(self, directory, data_transforms, image_datasets):
        # Save the checkpoint
        checkpoint = {'input_size': self.input_size,
                      'output_size': self.output_size,
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
