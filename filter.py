import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import matplotlib.pyplot as plt

import torchvision.datasets as dset
import torchvision.transforms as transforms

from PIL import Image
import string
import csv
import pickle
import glob
import os.path


class FilterNet(nn.Module):
    def __init__(self):
        super(FilterNet, self).__init__()

        # Define a convolutional layer
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
        # Define a rectified linear unit
        self.relu = torch.nn.ReLU()
        # Define a pooling layer
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Define another convolutional layer
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
        # We do not need to define model relus nor pooling (no parameters to train, we can reuse the same ones)

        # Define final fully-connected layers
        # self.fc1 = torch.nn.Linear(20 * 8 * 8, 120)
        self.fc1 = torch.nn.Linear(72000, 120)
        self.fc2 = torch.nn.Linear(120, 2)
        return

    def forward(self, x):
        # First stage: convolution -> relu -> pooling
        y = self.pool(self.relu(self.conv1(x)))
        # Second stage: convolution -> relu -> pooling
        y = self.pool(self.relu(self.conv2(y)))
        # Reshape to batch_size-by-whatever
        y = y.view(x.size(0), -1)
        # Last stage: fc -> relu -> fc
        y = self.fc2(self.relu(self.fc1(y)))
        # Return predictions
        return y


def get_net(retrain=False):
    filter_net = None
    if retrain is True:
        filter_net = train()
    else:
        filter_net = torch.load('filter_model.pth')

    return filter_net


def train(epochs=10):
    from filter import FilterNet

    # Create neural net
    filter_net = FilterNet()
    print(filter_net)

    # Create loss function & optimization criteria
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter_net.parameters(), lr=0.001, momentum=0.9)

    # Train network
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = dset.ImageFolder(root="datasets/training-filter/", transform=transform)
    train_loader = data_utils.DataLoader(trainset, batch_size=4, shuffle=True)

    for epoch in range(epochs):  # 3 iters
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):

            inputs, labels = data

            # Variable wrapper
            inputs, labels = Variable(inputs).float(), Variable(labels)

            # print(inputs, labels)
            optimizer.zero_grad()

            outputs = filter_net(inputs)
            # print(str(outputs), str(labels))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 20 == 19:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    print('\n\nFinished Training\n\n')

    torch.save(filter_net, 'filter_model.pth')
    return filter_net


def classify(retrain=False):
    filter_net = get_net(retrain)

    transform = transforms.Compose([transforms.ToTensor()])
    testset = dset.ImageFolder(root="datasets/testing-filter/", transform=transform)
    test_loader = data_utils.DataLoader(testset, batch_size=4, shuffle=True)

    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data
        outputs = filter_net(Variable(images))

        print(outputs.data)
        _, predicted = torch.max(outputs.data, 1)

        print('Predicted: ', ' '.join('%5s' % str(predicted[j][0])
                                      for j in range(len(predicted))))
        print('Labels: ' + str(labels))
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))


if __name__ == '__main__':
    classify(True)


