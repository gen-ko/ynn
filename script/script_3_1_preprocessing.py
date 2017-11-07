# required python version: 3.6+

import os
import sys
import src.load_data as load_data
from src import layer
import src.rbm as rbm
import src.autoencoder as autoencoder
import matplotlib.pyplot as plt
import numpy
import os

# Phase 0: Read File
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
data_filepath = '../data'
data_train_filename = 'train.txt'
data_valid_filename = 'val.txt'

data_train_filepath = os.path.join(path, data_filepath, data_train_filename)
data_valid_filepath = os.path.join(path, data_filepath, data_valid_filename)

# load text data from path
def load_from_path(data_filepath):
    # data_train: numpy.ndarray
    data_input = numpy.loadtxt(data_filepath, dtype=str, delimiter='\n')
    # x_train: numpy.ndarray
    return data_input
data_train = load_from_path(data_train_filepath)
data_valid = load_from_path(data_valid_filepath)


# Phase 1: split

# Create a vocabulary dictionary you are required to create an entry for every word in the training set
# also, make the data lower-cased
vocabulary: dict
all_lines: list = []
train_size = data_train.size
# will serve as a lookup table for the words and their corresponding id.
for i in range(train_size):
    all_lines.insert(i,  data_train[i].lower().split())

print(all_lines)






# Phase 2: add START tag and END tag :: TODO

# Phase 3: Compute the number of trainable parameters in the network :: TODO