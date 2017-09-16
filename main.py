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

def classify_image(path):
    slice_image(path)

    # Filter images (cell vs no cell)
    # Access split images in the temp/ directory

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = dset.ImageFolder(root=ROOT, transform=transform)
    data_loader = data_utils.DataLoader(dataset, shuffle=False)

    filter_net = filter.get_net()

    label_rows = []
    with open('data/labels.csv') as fd:
        reader = csv.reader(fd)
        label_rows = [row for row in reader]

    individual_labels = []

    for i, data in enumerate(data_loader, 0):
        images, labels = data

        outputs = filter_net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)

        row = int(i / 8) + 1
        col = int(i % 8) + 1

        fn_split = path.split('/')
        pathology_label = fn_split[len(fn_split) - 1].split('.')[0]

        img_path = pathology_label + ('_%02d' % row) + ('_%02d' % col) + '.png'

        if predicted.numpy()[0][0] == 0:  # 0 = no cells
            # helper.imshow(images)
            os.remove(BASE_TEMP_PATH + img_path)
        else:
            image_index = int(pathology_label.split('_')[0])
            # letter_ascii = string.ascii_uppercase.index(letter)
            # print(letter, letter_ascii)

            index = image_index * 64 + i

            print("IMAGE INDEX: " + str(image_index))
            print("i: " + str(i))

            label = int(label_rows[index][2])
            print(label)

            individual_labels.append(label)

            if not os.path.exists(BASE_TEMP_PATH + str(label)):
                os.makedirs(BASE_TEMP_PATH + str(label))

            os.rename(BASE_TEMP_PATH + img_path,
                      BASE_TEMP_PATH + str(label) + '/' + img_path)

    print(str(individual_labels))

    classifier_net = classifier.get_net()

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = dset.ImageFolder(root=BASE_TEMP_PATH, transform=transform)
    test_loader = data_utils.DataLoader(testset)

    numerator = 0
    denominator = 0

    predicted_counts = [0, 0, 0, 0]
    individual_predictions = []
    # individual_labels = []
    for data in test_loader:
        images, labels = data

        # helper.imshow(images)

        outputs = classifier_net(Variable(images))

        max, predicted = torch.max(outputs.data, 1)

        print('Predicted: ' + str(predicted))
        print(max[0].numpy())

        output_energies = outputs.data[0].numpy()

        print(output_energies)

        p = np.exp(max[0].numpy()) / np.sum(np.exp(output_energies))

        print(p)

        # print('Predicted: ', ' '.join('%5s' % str(predicted[j][0])
        #                               for j in range(len(predicted))))
        print('Labels: ' + str(labels))

        numerator += predicted[0].numpy()[0] * p[0]
        denominator += p[0]

        prediction_val = predicted[0].numpy()[0]

        predicted_counts[prediction_val] += 1
        individual_predictions.append(prediction_val)

        # true_label = labels[0]
        # print("TRUE LABEL: " + str(true_label))
        # individual_labels.append(true_label)



    print('Predicted Counts: ' + str(predicted_counts))
    weighted_avg = numerator / denominator
    print('Weighted average score for entire image: ' + str(weighted_avg))

    shutil.rmtree(BASE_TEMP_PATH)

    individual_labels.sort()
    return predicted_counts, weighted_avg, individual_predictions, individual_labels

if __name__ == '__main__': #ratio being used

    path = sys.argv[1]

    majority_votes = []
    weighted_avgs = []
    majority_counts = []
    predictions = []
    labels = []

    print(glob.glob('data/*.png'))

    shutil.rmtree(ROOT)
    os.makedirs(ROOT)

    # testing = [31, 4, 44, 27, 2, 22, 29, 9, 16, 43, 36, 15, 70, 68, 49, 34, 14, 53, 52, 10, 48, 67, 24, 8]
    testing = [1, 2, 3, 7, 8, 10, 14, 15, 16, 17, 18, 20]
    # training = [9, 22, 13, 11, 5, 19, 23, 21, 6, 12, 4, 0]

    for i, fn in enumerate(glob.glob('data/*.png'), 0):
        if i in testing:
            print(fn)
            majority_vote_prediction, weighted_avg, _predictions, _labels = classify_image(fn)
            print(majority_vote_prediction.index(max(majority_vote_prediction)))
            majority_votes.append(majority_vote_prediction.index(max(majority_vote_prediction)))
            weighted_avgs.append(weighted_avg)
            majority_counts.append(majority_vote_prediction)
            predictions.append(_predictions)
            labels.append(_labels)
        # else:
        #     majority_votes.append(-1)
        #     weighted_avgs.append(-1)

    print('MAJORITY VOTING PREDICTIONS: ' + str(majority_votes))
    print('MAJORITY VOTE COUNTS: ' + str(majority_counts))
    print('WEIGHTED AVERAGES: ' + str(weighted_avgs))

    print('INDIVIDUAL PREDICTIONS: ' + str(predictions))
    print('INDIVIDUAL LABELS: ' + str(labels))

    processed_labels = [] # using 30% rule
    for counts in majority_counts:
        if counts[3] / sum(counts) > 0.3:
            processed_labels.append(3)
        elif counts[2] / sum(counts) > 0.3:
            processed_labels.append(2)
        elif counts[1] / sum(counts) > 0.3:
            processed_labels.append(1)
        else:
            processed_labels.append(0)

    print('PROCESSED LABELS USING 30% RULE: ' + str(processed_labels))