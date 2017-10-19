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
import pickle

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


# x range [0, 1]
x_train, y_train = load_data.load_from_path(data_train_filepath)
x_valid, y_valid = load_data.load_from_path(data_valid_filepath)


#l1 = Layer(784, 100, 10)
print("start initiliazing...")

# SET UP GLOBAL PARAMETERS
lr = 0.05
momentum = 0.0
regularizer = 0.0

numpy.random.seed(1099)


layers = [layer.Linear(784, 100),
          layer.Sigmoid(100, 100),
          layer.Linear(100, 10),
          layer.Softmax(10, 10)]

myNN = NN(layers, learning_rate=lr, momentum=momentum, regularizer=regularizer)
full_path = os.path.realpath(__file__)
path, _ = os.path.split(full_path)
data_filepath = '../output/dump'
filepath = os.path.join(path, data_filepath, 'script-2-1-naive-autostop-rbm-whx-2639.dump')


with open(filepath, 'rb') as f:
    w, h_bias, x_bias = pickle.load(f)

myNN.layers[0].w = w.T
myNN.layers[0].b = h_bias

myNN.train(x_train, y_train, x_valid, y_valid, epoch=200, batch_size=32)

layers = [layer.Linear(784, 100),
          layer.Sigmoid(100, 100),
          layer.Linear(100, 10),
          layer.Softmax(10, 10)]
myNN = NN(layers, learning_rate=lr, momentum=momentum, regularizer=regularizer)
filepath = os.path.join(path, data_filepath, 'script-2-5-AE-autostop-rbm-whx-1579.dump')
with open(filepath, 'rb') as f:
    w, h_bias, x_bias = pickle.load(f)

myNN.layers[0].w = w.T
myNN.layers[0].b = h_bias

myNN.train(x_train, y_train, x_valid, y_valid, epoch=200, batch_size=32)

layers = [layer.Linear(784, 100),
          layer.Sigmoid(100, 100),
          layer.Linear(100, 10),
          layer.Softmax(10, 10)]
myNN = NN(layers, learning_rate=lr, momentum=momentum, regularizer=regularizer)
filepath = os.path.join(path, data_filepath, 'script-2-6-DAE-autostop-rbm-whx-1299.dump')
with open(filepath, 'rb') as f:
    w, h_bias, x_bias = pickle.load(f)
myNN.layers[0].w = w.T
myNN.layers[0].b = h_bias
myNN.train(x_train, y_train, x_valid, y_valid, epoch=200, batch_size=32)

layers = [layer.Linear(784, 100),
          layer.Sigmoid(100, 100),
          layer.Linear(100, 10),
          layer.Softmax(10, 10)]
myNN = NN(layers, learning_rate=lr, momentum=momentum, regularizer=regularizer)
myNN.train(x_train, y_train, x_valid, y_valid, epoch=200, batch_size=32)