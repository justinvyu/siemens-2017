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
import sys
import image_slicer
import filter
import classifier
import helper
import shutil


ROOT = 'temp/'
BASE_TEMP_PATH = ROOT + 'data/'

def slice_image(fn):
    try:
        img = Image.open(fn, 'r')
    except IOError:
        print("Image at specified path not found.")
        sys.exit()

    width, height = img.size  # Get dimensions
    left = (width - height) / 2
    right = (width + height) / 2

    # if image is less than 1944 in height, resize
    base_height = 1944
    if height is not base_height:
        hpercent = (base_height / float(height))
        new_width = (float(width) * float(hpercent))
        img = img.resize((new_width, base_height), Image.ANTIALIAS)

    img = img.crop((left, 0, right, base_height))

    fn_split = fn.split('/')

    if not os.path.exists(BASE_TEMP_PATH):
        os.makedirs(BASE_TEMP_PATH)

    new_img_path = BASE_TEMP_PATH + fn_split[len(fn_split) - 1]
    print(new_img_path)
    img.save(new_img_path, 'PNG')

    num_slices = 64
    image_slicer.slice(new_img_path, num_slices)

    os.remove(new_img_path)

if __name__ == '__main__':

    path = sys.argv[1]
    slice_image(path)

    # Filter images (cell vs no cell)
    # Access split images in the temp/ directory

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = dset.ImageFolder(root=ROOT, transform=transform)
    data_loader = data_utils.DataLoader(dataset, shuffle=False)

    filter_net = filter.get_net()

    label_rows = []
    with open('data/labels.csv') as fd:
        reader = csv.reader(fd)
        label_rows = [row for row in reader]

    for i, data in enumerate(data_loader, 0):
        images, labels = data

        outputs = filter_net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)

        row = int(i / 8) + 1
        col = int(i % 8) + 1

        fn_split = path.split('/')
        pathology_label = fn_split[len(fn_split) - 1].split('.')[0]

        img_path = pathology_label + ('_%02d' % row) + ('_%02d' % col) + '.png'
        print(img_path)

        if predicted.numpy()[0][0] == 0: # 0 = no cells
            # helper.imshow(images)
            os.remove(BASE_TEMP_PATH + img_path)
        else:
            letter = pathology_label[0]
            letter_ascii = string.ascii_uppercase.index(letter)
            print(letter, letter_ascii)

            index = letter_ascii * 64 + i
            print(index)

            label = int(label_rows[index][2])

            if not os.path.exists(BASE_TEMP_PATH + str(label)):
                os.makedirs(BASE_TEMP_PATH + str(label))

            os.rename(BASE_TEMP_PATH + img_path,
                      BASE_TEMP_PATH + str(label) + '/' + img_path)


    classifier_net = classifier.get_net()

    transform = transforms.Compose([transforms.ToTensor()])
    testset = dset.ImageFolder(root=BASE_TEMP_PATH, transform=transform)
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

    shutil.rmtree(BASE_TEMP_PATH)