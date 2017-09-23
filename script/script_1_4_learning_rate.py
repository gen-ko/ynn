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
network.init_nn(random_seed=20791)

random_state = numpy.random.RandomState(seed=2056791)



'''
myNN = NN(layers, learning_rate=0.01, debug=False, regularizer=0.001, momentum=0.9)


myNN.train_dump(x_train, y_train, x_valid, y_valid, epoch=200, dump_file=os.path.join(path, '../temp/network-2.dump'))
'''
'''
myNN = NN(layers, learning_rate=0.01, debug=False, regularizer=0.0001, momentum=0.9)


myNN.train_dump(x_train, y_train, x_valid, y_valid, epoch=200, dump_file=os.path.join(path, '../temp/network-3.dump'))

'''

hidden_units = [20, 100, 200, 500]
regularizers = [0.0, 0.0001, 0.001]
momentums = [0.0, 0.5, 0.9]
learning_rates = [0.1, 0.01, 0.2, 0.5]




for i1 in range(len(hidden_units)):
    for i2 in range(len(regularizers)):
        for i3 in range(len(momentums)):
            for i4 in range(len(learning_rates)):
                layers = [layer.Linear(784, 100),
                          layer.BN(100, 100),
                          layer.ReLU(100, 100),
                          layer.BN(100, 100),
                          layer.SoftmaxLayer(100, 10)]
                name = 'network-' + str(i1) + '-' + str(i2) + '-' + str(i3) + '-' + str(i4) + '.dump'
                myNN = NN(layers, learning_rate=learning_rates[i4], regularizer=regularizers[i2], momentum=momentums[i3])
                myNN.train(x_train, y_train, x_valid, y_valid, epoch=300, batch_size=32)


print('training finished')
