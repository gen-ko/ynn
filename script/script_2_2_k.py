# required python version: 3.6+

import os
import sys
import src.load_data as load_data
from src import layer
import src.rbm as rbm
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
#rbm.init_rbm(random_seed=1099)
#numpy.random.RandomState(seed=1099)
numpy.random.seed(5800)

x_train, y_train = load_data.load_from_path(data_train_filepath)
x_valid, y_valid = load_data.load_from_path(data_valid_filepath)

myRBM = rbm.RBM(28*28, hidden_units=100)
myRBM.set_visualize(28, 28, stride=20)
myRBM.set_plot(stride=20)
myRBM.set_autostop(window=80, stride=40)
myRBM.train(x_train, x_valid, k=5, epoch=3000, learning_rate=0.05, batch_size=32, plotfile='script-2-2-k=5')

myRBM = rbm.RBM(28*28, hidden_units=100)
myRBM.set_visualize(28, 28, stride=20)
myRBM.set_plot(stride=20)
myRBM.set_autostop(window=80, stride=40)
myRBM.train(x_train, x_valid, k=10, epoch=3000, learning_rate=0.05, batch_size=32, plotfile='script-2-2-k=10')

myRBM = rbm.RBM(28*28, hidden_units=100)
myRBM.set_visualize(28, 28, stride=20)
myRBM.set_plot(stride=20)
myRBM.set_autostop(window=80, stride=40)
myRBM.train(x_train, x_valid, k=20, epoch=3000, learning_rate=0.05, batch_size=32, plotfile='script-2-2-k=20')