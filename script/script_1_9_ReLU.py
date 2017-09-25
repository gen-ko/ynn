# required python version: 3.6+

import os
import sys
import src.load_data as load_data
from src import plot_data
from src import layer
from src.network import NeuralNetwork_Dumpable as NN
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


network.init_nn(random_seed=1099)


x_train, y_train = load_data.load_from_path(data_train_filepath)
x_valid, y_valid = load_data.load_from_path(data_valid_filepath)

is_ReLU = False
is_Tanh = (not is_ReLU ) and True

if is_ReLU:
    print('start initializing ReLU...')
    layers = [layer.Linear(784, 100),
              layer.ReLU(100, 100),
              layer.Linear(100, 100),
              layer.Softmax(10, 10)]

    myNN = NN(layers, learning_rate=0.1, regularizer=0.0001, momentum=0.9)
    myNN.train(x_train, y_train, x_valid, y_valid, epoch=300, batch_size=32)
elif is_Tanh:
    print('start initializing Tanh...')
    layers = [layer.Linear(784, 100),
              layer.Tanh(100, 100),
              layer.Linear(100, 100),
              layer.Softmax(10, 10)]

    myNN = NN(layers, learning_rate=0.1, regularizer=0.0001, momentum=0.9)
    myNN.train(x_train, y_train, x_valid, y_valid, epoch=300, batch_size=32)
else:
    print('start initializing Sigmoid...')
    layers = [layer.Linear(784, 100),
              layer.Sigmoid(100, 100),
              layer.Linear(100, 100),
              layer.Softmax(10, 10)]

    myNN = NN(layers, learning_rate=0.1, regularizer=0.0001, momentum=0.9)
    myNN.train(x_train, y_train, x_valid, y_valid, epoch=300, batch_size=32)