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
network.init_nn(random_seed=2056791)

random_state = numpy.random.RandomState(seed=2056791)


layers = [layer.SigmoidLayer(784, 100),
          layer.SoftmaxLayer(100, 10)]

myNN = NN(layers, learning_rate=0.01, debug=False, num_class=10)


myNN.train_trace(x_train, y_train, x_valid, y_valid, epoch=200)
#score = myNN.score(x_valid, y_valid)
#myNN2 = layer.MultiLayerNetwork(784, 10, learning_rate=0.001)
#myNN2.train(x_train, y_train, x_valid, y_valid, epoch=100)
#myNN3 = layer.SingleLayerNetwork(784, 100, 10, debug=False, learning_rate=0.2)
#myNN3.train_setting(epoch=200, learning_rate=0.01, batch_size=128, weight_decay=0.001, momentum=0.9)
#myNN3.train_in_batch(x_train, y_train, x_valid, y_valid)
print("hi")