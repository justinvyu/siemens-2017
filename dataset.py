import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import string
import csv
import pickle
import glob

if __name__ == '__main__':

    # np.set_printoptions(threshold=np.nan)

    generic_path = './data/'
    # data_path = 'S (6149).jpg'

    label_rows = []
    with open(generic_path + 'labels.csv') as fd:
        reader = csv.reader(fd)
        label_rows = [row for row in reader]

    for fn in glob.glob(generic_path + '*.jpg'):
        img = Image.open(fn).convert('RGB')
        data_path = fn.split('/')[2]
        img_arr = np.array(img)

        num_rows = 8
        num_cols = 8

        letter_index = string.ascii_uppercase.index(data_path[0])
        label_start_index = letter_index * num_rows * num_cols

        full_width = img_arr.shape[0]
        full_height = img_arr.shape[1]

        slice_width = full_width // num_cols
        slice_height = full_height // num_rows

        slices = []
        index = label_start_index

        for row in range(num_rows):
            startRow = row * slice_height
            endRow = (row + 1) * slice_height

            for col in range(num_cols):
                startCol = col * slice_width
                endCol = (col + 1) * slice_height

                labeled = (img_arr[startRow:endRow, startCol:endCol], label_rows[index][2])
                index += 1
                slices.append(labeled)

        print(len(slices))
        with open('data.pickle', 'wb') as handle:
            pickle.dump(slices, handle)

            # print(img_arr) # 1944 by 1944 by depth=3 (RGB)
