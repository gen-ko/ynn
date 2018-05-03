# required python version: 3.6+

import os
from src import layer
from src.network import DNN
import numpy
import tensorflow as tf

#  format of data
# disitstrain.txt contains 3000 lines, each line 785 numbers, comma delimited

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
data_filepath = '../data'
data_train_filename = 'digitstrain.txt'
data_valid_filename = 'digitsvalid.txt'
data_test_filename = 'digitstest.txt'



mnist = tf.contrib.learn.datasets.load_dataset("mnist")





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
# network.init_nn(random_seed=1099)
x_train = mnist.train.images # 55000 x 784
x_valid = mnist.validation.images # 5000 x 784

y_train = mnist.train.labels
y_valid = mnist.validation.labels

layers = [layer.Linear(784, 100),
          #layer.BN(100, 100),
          layer.Sigmoid(100),
          layer.Linear(100, 10),
          layer.Softmax(10)]

myNN = DNN(layers, learning_rate=0.01, momentum=0.90, regularizer=0.0001)

myNN.train(x_train, y_train, x_valid, y_valid, epoch=300, batch_size=64)
print("hi")