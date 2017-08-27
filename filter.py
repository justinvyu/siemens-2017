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


label_csv_path = 'labels.csv'
dataset_path = 'filter_data.pickle'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Define a convolutional layer
        self.conv1 = torch.nn.Conva2d(3, 10, kernel_size=3, stride=1, padding=1)
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

def create_dataset():
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Scale((243, 243)),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform = transforms.Compose(
        [transforms.ToTensor()])
    trainset = dset.ImageFolder(root="training", transform=transform)

# def create_dataset():
#     generic_path = './data/'
#
#     label_rows = []
#     with open(generic_path + label_csv_path) as fd:
#         reader = csv.reader(fd)
#         label_rows = [row for row in reader]
#
#     inputs = []
#     labels = []
#     for fn in glob.glob(generic_path + '*.jpg'):
#         img = Image.open(fn).convert('RGB')
#         data_path = fn.split('/')[2]
#         img_arr = np.array(img)
#
#         num_rows = 8
#         num_cols = 8
#
#         letter_index = string.ascii_uppercase.index(data_path[0])
#         label_start_index = letter_index * num_rows * num_cols
#
#         full_width = img_arr.shape[0]
#         full_height = img_arr.shape[1]
#
#         slice_width = full_width // num_cols
#         slice_height = full_height // num_rows
#
#         index = label_start_index
#
#         for row in range(num_rows):
#             startRow = row * slice_height
#             endRow = (row + 1) * slice_height
#
#             for col in range(num_cols):
#                 startCol = col * slice_width
#                 endCol = (col + 1) * slice_height
#
#                 # 0 is <=50% cell area
#                 # 1 is >50% cell area
#
#                 label = 0
#                 percent = int(label_rows[index][3].replace('%', ''))
#                 if percent > 50:
#                     label = 1
#
#                 inputs.append(img_arr[startRow:endRow, startCol:endCol])
#                 labels.append(label)
#
#                 index += 1
#
#     print(np.array(inputs).shape)
#     # num_batches = 12
#
#     # batch_len = len(labels) // num_batches
#     # batch_labels = list(chunks(labels, batch_len))
#     # batch_inputs = list(chunks(inputs, batch_len))
#
#     # data = np.array(batch_inputs), np.array(batch_labels)
#
#     # data = np.array(inputs).reshape(1536, 3, 243, 243), np.array(labels)
#     data = np.array(inputs).reshape(1536, 243, 243, 3), np.array(labels)
#
#     with open(dataset_path, 'wb') as handle:
#         pickle.dump(data, handle)


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

if __name__ == '__main__':
    # if not os.path.isfile(dataset_path):
    #     create_dataset()
    #
    # data_loader = None
    # with open(dataset_path, 'rb') as handle:
    #     data_loader = pickle.load(handle)

    # Create neural net
    net = Net()
    print(net)

    # Create loss function & optimization criteria
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train network

    # batch_inputs, batch_labels = data_loader

    # train = data_utils.TensorDataset(torch.from_numpy(batch_inputs), torch.from_numpy(batch_labels))
    # train_loader = data_utils.DataLoader(train, batch_size=4, shuffle=True)

    transform = transforms.Compose(
        [transforms.ToTensor()
         ])
    trainset = dset.ImageFolder(root="training/", transform=transform)
    train_loader = data_utils.DataLoader(trainset, batch_size=4, shuffle=True)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    show_grid(images, 2)

    for epoch in range(3): # 3 iters
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):

            inputs, labels = data

            # Variable wrapper
            inputs, labels = Variable(inputs).float(), Variable(labels)

            # print(inputs, labels)
            optimizer.zero_grad()

            outputs = net(inputs)
            # print(str(outputs), str(labels))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    print('\n\nFinished Training\n\n')

    test_loader = data_utils.DataLoader(trainset, batch_size=16, shuffle=True)
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    show_grid(images, 4)

    test_outputs = net(Variable(images).float())
    _, predicted = torch.max(test_outputs, 1)
    # print('Predicted: ', ' '.join('%5s' % str(predicted[j][0].data[0])
    #                               for j in range(16)))
    print('Predicted: ' + str(test_outputs))

    print('Labels: ' + str(labels))
