import image_slicer
from PIL import Image
import sys
import csv
import glob
import string
import numpy as np
import random

def cut_images(path):
    print(img_path)
    for fn in glob.glob(img_path + '/*.png'):
        try:
            img = Image.open(fn, 'r')
        except IOError:
            print("Image at specified path not found.")
            sys.exit()

        width, height = img.size   # Get dimensions
        left = (width - height) / 2
        right = (width + height) / 2

        # if image is less than 1944 in height, resize
        base_height = 1944
        if height is not base_height:
            hpercent = (base_height / float(height))
            new_width = (float(width) * float(hpercent))
            img = img.resize((new_width, base_height), Image.ANTIALIAS)

        img = img.crop((left, 0, right, base_height))

        new_img_path = 'processed_8x8/' + fn.split('/')[1]
        print(new_img_path)
        img.save(new_img_path, 'PNG')

        num_slices = 64
        image_slicer.slice(new_img_path, num_slices)

def label_images_filter(path):
    label_rows = []
    with open('labels.csv') as fd:
        reader = csv.reader(fd)
        label_rows = [row for row in reader]

    for fn in glob.glob(path + '/*.png'):
        remove_suffix = fn.split('.png')[0]
        row = int(remove_suffix.split('_')[2])
        col = int(remove_suffix.split('_')[3])

        letter = fn.split('/')[1][0]
        letter_ascii = string.ascii_uppercase.index(letter)

        index = ((row - 1) * 8 + col) - 1
        index = letter_ascii * 64 + index

        label = 0
        percent = int(label_rows[index][3].replace('%', ''))
        if percent > 50:
            label = 1

        img = Image.open(fn, 'r')
        img.save('labeled_data_filter/' + str(label) + '_' + fn.split('/')[1])


def label_images_dataset(path):

    np.random.seed(100)

    # training_testing_decider = np.arange(26)
    # np.random.shuffle(training_testing_decider)
    # training_indices = training_testing_decider[:13]
    # testing_indices = training_testing_decider[13:]

    # print(training_indices)
    # print(testing_indices)

    label_rows = []
    with open('labels.csv') as fd:
        reader = csv.reader(fd)
        label_rows = [row for row in reader]

    num_images = len(label_rows)

    testing_ct = 0
    training_ct = 0

    for fn in glob.glob(path + '/*.png'):
        remove_suffix = fn.split('.png')[0]
        row = int(remove_suffix.split('_')[4])
        col = int(remove_suffix.split('_')[5])

        letter = fn.split('/')[2][2]
        letter_ascii = string.ascii_uppercase.index(letter)

        index = ((row - 1) * 8 + col) - 1
        index = letter_ascii * 64 + index

        label = int(label_rows[index][2])

        subdir_path = ''
        # if letter_ascii in training_indices:
        #     subdir_path = 'training/'
        # elif letter_ascii in testing_indices:
        #     subdir_path = 'testing/'

        rand = random.randint(0, 1)
        if training_ct == num_images:
            subdir_path = 'testing/'
        elif testing_ct == num_images:
            subdir_path = 'training/'
        elif rand == 1:
            subdir_path = 'testing/'
        else:
            subdir_path = 'training/'

        img = Image.open(fn, 'r')
        img.save('labeled_data_split/' + subdir_path + str(label) + '/' + str(label) + '_' + fn.split('/')[2])

if __name__ == '__main__':

    img_path = sys.argv[1]

    label_images_dataset(img_path)
