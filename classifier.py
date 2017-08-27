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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

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
        self.fc2 = torch.nn.Linear(120, 4) # 0, 1, 2, or 3
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


def imshow(img):
    npimg = img.numpy().reshape(243, 243, 3)
    plt.imshow(npimg)
    plt.show()


def show_grid(images, size):
    fig = plt.figure()
    for i in range(size * size):
        fig.add_subplot(size, size, i + 1)
        plt.imshow(images[i].numpy().reshape(243, 243, 3))
    plt.show()


def train(epochs=20):
    # Create neural net
    net = Net()
    print(net)

    # Create loss function & optimization criteria
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0006, momentum=0.9)

    # Train network
    transform = transforms.Compose(
        [transforms.ToTensor()
         ])
    trainset = dset.ImageFolder(root="datasets/training-classifier/", transform=transform)
    train_loader = data_utils.DataLoader(trainset, batch_size=4, shuffle=True)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            # Variable wrapper
            inputs, labels = Variable(inputs).float(), Variable(labels)
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    print('\n\nFinished Training\n\n')

    torch.save(net, 'model.pth')

    return net


def classify():
    net = train()

    transform = transforms.Compose(
        [transforms.ToTensor()
         ])
    testset = dset.ImageFolder(root="datasets/testing-classifier/", transform=transform)
    test_loader = data_utils.DataLoader(testset, batch_size=4, shuffle=True)

    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data
        outputs = net(Variable(images))

        print(outputs.data)
        _, predicted = torch.max(outputs.data, 1)

        print('Predicted: ', ' '.join('%5s' % str(predicted[j][0])
                                      for j in range(len(predicted))))
        print('Labels: ' + str(labels))
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))


def classify_weighted_avg(retrain=False):
    net = None
    if retrain is True:
        net = train()
    else:
        net = torch.load('model.pth')

    transform = transforms.Compose(
        [transforms.ToTensor()
         ])
    testset = dset.ImageFolder(root="datasets/testing-classifier-u/", transform=transform)
    test_loader = data_utils.DataLoader(testset)

    numerator = 0
    denominator = 0

    predicted_counts = [0, 0, 0, 0]
    for data in test_loader:
        images, labels = data
        outputs = net(Variable(images))

        max, predicted = torch.max(outputs.data, 1)

        print('Predicted: ' + str(predicted))
        output_energies = outputs.data[0].numpy()

        p = np.exp(max[0].numpy()) / np.sum(np.exp(output_energies))
        print(p)
        print('Labels: ' + str(labels))

        numerator += predicted[0].numpy()[0] * p[0]
        denominator += p[0]
        predicted_counts[predicted[0].numpy()[0]] += 1

    print('Predicted Counts: ' + str(predicted_counts))
    weighted_avg = numerator / denominator
    print('Weighted average score for entire image: ' + str(weighted_avg))


if __name__ == '__main__':
    classify_weighted_avg()
