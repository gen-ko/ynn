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


# x range [0, 1]
x_train, y_train = load_data.load_from_path(data_train_filepath)
x_valid, y_valid = load_data.load_from_path(data_valid_filepath)

# warm-up phase
'''
print(x_train.shape)
print(y_train.shape)
print(y_train)

x_train_reshaped = plot_data.reshape_row_major(x_train[2550], 28, 28)
# plt.imshow(x_train_reshaped)

# so data is row majored
plot_data.plot_image(x_train_reshaped)
plt.show()
'''
#l1 = Layer(784, 100, 10)
print("start initiliazing...")
network.init_nn(random_seed=1099)


layers = [layer.Linear(784, 500),
          layer.Sigmoid(500, 500),
          layer.Linear(500, 500),
          layer.Sigmoid(500, 500),
          layer.SoftmaxLayer(100, 10)]

myNN = NN(layers, learning_rate=0.01, debug=False, momentum=0.5, regularizer=0.0001)

myNN.train(x_train, y_train, x_valid, y_valid, epoch=300)
print("hi")