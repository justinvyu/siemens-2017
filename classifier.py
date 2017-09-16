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
from PIL import ImageFile
import string
import csv
import pickle
import glob
import os.path
import filter
import helper
import visualize

class ClassifierNet(nn.Module):
    def __init__(self):
        super(ClassifierNet, self).__init__()

        # Define a convolutional layer
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
        # Define a rectified linear unit
        self.relu = torch.nn.ReLU()
        # Define a pooling layer
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Define another convolutional layer
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
        # We do not need to define model relus nor pooling (no parameters to train, we can reuse the same ones)

        # self.conv2_drop = nn.Dropout2d()

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
        # y = F.dropout(y, training=self.training)
        y = self.fc2(self.relu(self.fc1(y)))
        # Return predictions
        return y


def train(epochs=20, continue_training=False):
    from classifier import ClassifierNet

    # Create neural net
    net = ClassifierNet()
    if continue_training is True:
        net = get_net()

    print(net)

    # Create loss function & optimization criteria
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0008, momentum=0.9)

    # Train network
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = dset.ImageFolder(root="datasets/training-classifier-RATIO/", transform=transform)
    train_loader = data_utils.DataLoader(trainset, batch_size=6, shuffle=True)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    loss_over_time_avgs = []
    loss_over_time = []

    for epoch in range(epochs):

        loss_over_time = []

        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):

            inputs, labels = data

            # Variable wrapper
            inputs, labels = Variable(inputs).float(), Variable(labels)

            # print(inputs, labels)
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 20 == 19:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))

                loss_over_time.append(running_loss / 20)
                running_loss = 0.0

        loss_over_time_avgs.append(np.mean(np.array(loss_over_time)))


    print(str(loss_over_time_avgs))
    plt.plot(loss_over_time_avgs)
    plt.xlabel('Iteration')
    plt.ylabel('Running Loss of Classifier')
    plt.show()

    print('\n\nFinished Training\n\n')

    torch.save(net, 'model.pth')
    return net


def get_net(retrain=False):
    classifier_net = None
    if retrain is True:
        classifier_net = train()
    else:
        classifier_net = torch.load('model.pth')

    return classifier_net

def classify():
    net = get_net()

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = dset.ImageFolder(root="old_datasets/testing-classifier-split-noZero/", transform=transform)
    test_loader = data_utils.DataLoader(testset, batch_size=4, shuffle=True)

    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data
        outputs = net(Variable(images))

        # helper.show_grid(images, 2)

        print(outputs.data)
        _, predicted = torch.max(outputs.data, 1)

        print('Predicted: ', ' '.join('%5s' % str(predicted[j][0])
                                      for j in range(len(predicted))))
        print('Labels: ' + str(labels))
        total += labels.size(0)
        correct += (predicted == labels).sum()

        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))

        # helper.show_grid(images, 6)

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

def classify_weighted_avg(retrain=False):
    classifier_net = get_net(retrain)

    transform = transforms.Compose([transforms.ToTensor()])
    testset = dset.ImageFolder(root="datasets/testing-classifier/", transform=transform)
    test_loader = data_utils.DataLoader(testset)

    numerator = 0
    denominator = 0

    predicted_counts = [0, 0, 0, 0]
    for data in test_loader:
        images, labels = data
        outputs = classifier_net(Variable(images))

        max, predicted = torch.max(outputs.data, 1)

        print('Predicted: ' + str(predicted))
        # print(max[0].numpy())

        output_energies = outputs.data[0].numpy()

        # print(output_energies)

        p = np.exp(max[0].numpy()) / np.sum(np.exp(output_energies))

        print(p)

        # print('Predicted: ', ' '.join('%5s' % str(predicted[j][0])
        #                               for j in range(len(predicted))))
        print('Labels: ' + str(labels))

        numerator += predicted[0].numpy()[0] * p[0]
        denominator += p[0]

        predicted_counts[predicted[0].numpy()[0]] += 1

    print('Predicted Counts: ' + str(predicted_counts))
    weighted_avg = numerator / denominator
    print('Weighted average score for entire image: ' + str(weighted_avg))

if __name__ == '__main__':
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    train(10)
