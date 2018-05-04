# required python version: 3.6+

import src.nlp as nlp
import numpy
import os
import pickle
from src import callback as cb
from src import train as utf

from src.util.status import DataStore
from src.util.status import TrainSettings
from src import network
from src import layer

import tensorflow as tf





# set the random seed
numpy.random.seed(1099)


mnist = tf.contrib.learn.datasets.load_dataset("mnist")
x_train = mnist.train.images # 55000 x 784
x_valid = mnist.validation.images # 5000 x 784

y_train = mnist.train.labels
y_valid = mnist.validation.labels

data_store_train = DataStore(x_train, y_train)
data_store_valid = DataStore(x_valid, y_valid)

#############################
train_settings = TrainSettings(learning_rate=0.01, batch_size=16, momentum=0.0, plot_callback=cb.plot_callback,
                                  loss_callback=cb.loss_callback, filename='script-4-1', epoch=100, prefix='e16')

layers = [layer.ProbabilisticGaussianLinear(784, 100),
          #layer.BN(100, 100),
          layer.Sigmoid(100),
          layer.ProbabilisticGaussianLinear(100, 10),
          layer.Softmax(10)]

mynn = network.MLP(layers)

utf.cross_train(mynn, data_store_train, data_store_valid, train_settings)
