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
path, _ = os.path.split(os.path.realpath(__file__))
dictionary_filename = os.path.join(path,'../output/dump', 'dictionary.dump')


# phase A, train the model
train = False
if train:
    # load data
    data_train_filepath = os.path.join(path, '../output/dump', 'train.dump')
    data_valid_filepath = os.path.join(path, '../output/dump', 'valid.dump')

    # load preprocessed data
    with open(data_train_filepath, 'rb+') as f:
        data_train = pickle.load(f)
    with open(data_valid_filepath, 'rb+') as f:
        data_valid = pickle.load(f)

    x_train = data_train[:, 0:3]
    y_train = data_train[:, 3]
    x_valid = data_valid[:, 0:3]
    y_valid = data_valid[:, 3]

    data_store_train = uf.DataStore(x_train, y_train)
    data_store_valid = uf.DataStore(x_valid, y_valid)

    train_settings = uf.TrainSettings(learning_rate=0.1, batch_size=512, momentum=0.0, plot_callback=cb.plot_callback,
                                      loss_callback=cb.loss_callback, filename='script-3-5', epoch=100, prefix='h128')
    # build the neural network
    mynlp = nlp.NlpL3TypeA(dict_size=8000, embedding_size=2, hidden_units=128)
    utf.cross_train(mynlp, data_store_train, data_store_valid, train_settings)

else:
    # load the network
    nn_filename = os.path.join(path, '../output/dump', 'script-3-5-h128-i-s.dump')
    nn = nlp.NlpL3TypeA(8000, 2, 128)
    with open(nn_filename, 'rb') as f:
        nn.load(pickle.load(f))

    # load vocabulary
    with open(dictionary_filename, 'rb') as f:
        d: dict = pickle.load(f)
    # visualize the words

    # draw 500 random words from d
    dv = list(d.values())
    dk = list(d.keys())
    dk_random = numpy.array(dk)
    d_random = numpy.random.choice(dv, size=(500, ))
    d_list = list(d_random)
    weights: dict = nn.layers[0].w
    y = weights[d_random]
    plt.scatter(y[:, 0], y[:, 1])
    plt.show()

    dk_random = dk_random[d_list]

    for label, xi, yi in zip(dk_random, y[:, 0], y[:, 1]):
        plt.annotate(label, xy=(xi, yi), xytext=(0, 0), textcoords='offset points')
    plt.xlim(min(y[:, 0]), max(y[:, 0]))
    plt.ylim(min(y[:, 1]), max(y[:, 1]))
    plt.show()
    print('hi')
