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


# x range [0, 1]
x_train, y_train = load_data.load_from_path(data_train_filepath)
x_valid, y_valid = load_data.load_from_path(data_valid_filepath)
print("start initiliazing...")
network.init_nn(random_seed=20791)


layers = [layer.Sigmoid(784, 100),
          layer.Softmax(100, 10)]

myNN = NN(layers)
myNN.set_training(learning_rate=0.01, regularizer=0.0001, momentum=0.9, batch_size=128, epoch=200)
myNN.feed_train(x_train, y_train)
myNN.train()


print("hi")