# required python version: 3.6+

import os
import sys
import src.load_data as load_data
from src import layer
import src.nlp as nlp
import matplotlib.pyplot as plt
import numpy
import os
import pickle
from src import util as uf
from src import callback as cb
from src import train as utf


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
data_store_train = uf.DataStore(x_train, y_train)
data_store_valid = uf.DataStore(x_valid, y_valid)


train_settings = uf.TrainSettings(learning_rate=0.1, batch_size=512, momentum=0.0, plot_callback=cb.plot_callback,
                                  loss_callback=cb.loss_callback, filename='script-3-3', epoch=100, prefix='h128')
# build the neural network
mynlp = nlp.NlpL3TypeB(dict_size=8000, embedding_size=16, hidden_units=128)
utf.cross_train(mynlp, data_store_train, data_store_valid, train_settings)
