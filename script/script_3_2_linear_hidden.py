# required python version: 3.6+

import os
import sys
import src.load_data as load_data
from src import layer
from src import network
import src.nlp as nlp
import matplotlib.pyplot as plt
import numpy
import os
import pickle


# resolve file path
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
dump_filepath = '../output/dump'
data_train_filename = 'train.dump'
data_valid_filename = 'valid.dump'

data_train_filepath = os.path.join(path, dump_filepath, data_train_filename)
data_valid_filepath = os.path.join(path, dump_filepath, data_valid_filename)


# load preprocessed data
with open(data_train_filepath, 'rb+') as f:
    data_train = pickle.load(f)
with open(data_valid_filepath, 'rb+') as f:
    data_valid = pickle.load(f)

x_train = data_train[:, 0:3]
y_train = data_train[:, 3]
x_valid = data_valid[:, 0:3]
y_valid = data_valid[:, 3]


# set the random seed
numpy.random.seed(1099)


layers = [layer.Embedding(8000, 16, 'eb-1'),
          layer.Embedding(8000, 16, 'eb-2'),
          layer.Embedding(8000, 16, 'eb-3'),
          layer.Linear(16, 128, 'l1-1'),
          layer.Linear(16, 128, 'l1-2'),
          layer.Linear(16, 128, 'l1-3'),
          layer.Linear(128, 8000, 'l2'),
          layer.Softmax(8000, 'softmax')]

connection = {'input': ['eb-1', 'eb-2', 'eb-3'],
              'eb-1': ['l1-1'],
              'eb-2': ['l1-2'],
              'eb-3': ['l1-3'],
              'l1-1': ['l2'],
              'l1-2': ['l2'],
              'l1-3': ['l2'],
              'l2': ['softmax'],
              'output': ['softmax']
              }

myNN = network.NeuralNetwork(layers, connection)





# build the neural network
mynlp = nlp.NlpL3TypeA()
mynlp.train(x_train, y_train, x_valid, y_valid, epoch=100, batch_size=512, momentum=0.9)
print(type(x_train))