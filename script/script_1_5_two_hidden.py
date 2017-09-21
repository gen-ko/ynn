# required python version: 3.6+

import os
import sys
import src.load_data as load_data
from src import plot_data
from src import layer
from src.network import NeuralNetwork as NN
import src.network as network
import matplotlib.pyplot as plt
import numpy
import os

#  format of data
# disitstrain.txt contains 3000 lines, each line 785 numbers, comma delimited

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
data_filepath = '../data'
data_train_filename = 'digitstrain.txt'
data_valid_filename = 'digitsvalid.txt'
data_test_filename = 'digitstest.txt'

data_train_filepath = os.path.join(path, data_filepath, data_train_filename)
data_valid_filepath = os.path.join(path, data_filepath, data_valid_filename)
data_test_filepath = os.path.join(path, data_filepath, data_test_filename)

print('start initializing...')
network.init_nn(random_seed=20791)

learning_rates = [0.01]
momentums = [0.9]

regularizers = [0.0001]
x_train, y_train = load_data.load_from_path(data_train_filepath)
x_valid, y_valid = load_data.load_from_path(data_valid_filepath)

for i2 in range(len(regularizers)):
    for i3 in range(len(momentums)):
        for i4 in range(len(learning_rates)):
            layers = [layer.SigmoidLayer(784, 100),
                      layer.SigmoidLayer(100, 100),
                      layer.SoftmaxLayer(100, 10)]
            name = 'network2' + '-' + str(i2) + '-' + str(i3) + '-' + str(i4) + '.dump'
            myNN = NN(layers, learning_rate=learning_rates[i4], regularizer=regularizers[i2], momentum=momentums[i3])
            myNN.train_dump(x_train, y_train, x_valid, y_valid, epoch=300,
                            dump_file=os.path.join(path, '../temp', name))