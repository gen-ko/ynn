import numpy
import pickle
from src import layer
import os
from time import gmtime, strftime
import matplotlib.pyplot as plt
from src import util as uf
from src import network as nn
from src import plotter


class NlpL3TypeA(object):
    def __init__(self, layers: [layer.Layer]):



    def fprop(self, x, keep_state: bool=False):
        if keep_state:
            self.h0_0 = x[:, 0]
            self.h0_1 = x[:, 1]
            self.h0_2 = x[:, 2]
            self.h1_0 = self.layer0.forward(self.h0_0)
            self.h1_1 = self.layer0.forward(self.h0_1)
            self.h1_2 = self.layer0.forward(self.h0_2)
            self.h2_0 = self.layer1_0.forward(self.h1_0)
            self.h2_1 = self.layer1_1.forward(self.h1_1)
            self.h2_2 = self.layer1_2.forward(self.h1_2)
            self.h2 = self.h2_0 + self.h2_1 + self.h2_2
            self.h3 = self.layer2.forward(self.h2)
            self.h4 = self.layer3.forward(self.h3)
            return self.h4
        else:
            h0_0 = x[:, 0]
            h0_1 = x[:, 1]
            h0_2 = x[:, 2]
            h1_0 = self.layer0.forward(h0_0)
            h1_1 = self.layer0.forward(h0_1)
            h1_2 = self.layer0.forward(h0_2)
            h2_0 = self.layer1_0.forward(h1_0)
            h2_1 = self.layer1_1.forward(h1_1)
            h2_2 = self.layer1_2.forward(h1_2)
            h2 = h2_0 + h2_1 + h2_2
            h3 = self.layer2.forward(h2)
            h4 = self.layer3.forward(h3)
            return h4

    def bprop(self, y):
        d_h3 = self.layer3.backward(y, self.h4, self.h3)
        d_h2 = self.layer2.backward(d_h3, self.h3, self.h2)
        # the gradiants of h2 is the same as h2_0, h2_1 and h2_2, which can be derived
        d_h1_2 = self.layer1_2.backward(d_h2, self.h2_2, self.h1_2)
        d_h1_1 = self.layer1_1.backward(d_h2, self.h2_1, self.h1_1)
        d_h1_0 = self.layer1_0.backward(d_h2, self.h2_0, self.h1_0)
        self.layer0.backward(d_h1_2, self.h1_2, self.h0_2)
        self.layer0.backward(d_h1_1, self.h1_1, self.h0_1)
        self.layer0.backward(d_h1_0, self.h1_0, self.h0_0)
        return

    def update(self):
        self.layer0.update(self.learning_rate)

        self.layer1_0.update(self.learning_rate, momentum=self.momentum)
        self.layer1_1.update(self.learning_rate, momentum=self.momentum)
        self.layer1_2.update(self.learning_rate, momentum=self.momentum)

        self.layer2.update(self.learning_rate, momentum=self.momentum)
        return



    def train(self, x_train, y_train, x_valid, y_valid, epoch=200, batch_size=256, learning_rate=0.1, momentum=0.0):
        print('------------------ Start Training -----------------')
        print('\tepoch\t|\ttrain loss\t|\ttrain error\t|\tvalid loss\t|\tvalid error\t')
        self.momentum = momentum
        self.train_error = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        self.valid_error = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        self.train_loss = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        self.valid_loss = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        self.batch_size = batch_size
        self.learning_rate = learning_rate


        for j in range(epoch):
            x_train, y_train = uf.shuffle(x_train, y_train)
            train_score = 0.0
            for i in range(0, y_train.size, batch_size):
                x_batch = x_train[i: i + batch_size]
                y_batch = y_train[i: i + batch_size]

                output_train = self.fprop(x_batch, keep_state=True)
                predict_train = uf.pick_class(output_train)
                self.train_loss[j] += uf.cross_entropy_loss(output_train, y_batch, take_average=False)

                train_score += numpy.sum(predict_train == y_batch)
                self.bprop(y_batch)
                self.update()

            self.train_loss[j] = self.train_loss[j] / y_train.shape[0]
            self.train_error[j] = (1.0 - train_score / y_train.shape[0])

            # start a validation
            output_valid = self.fprop(x_valid, keep_state=False)
            predict_valid = uf.pick_class(output_valid)
            self.valid_loss[j] = uf.cross_entropy_loss(output_valid, y_valid, take_average=True)
            valid_score = numpy.sum(predict_valid == y_valid)
            self.valid_error[j] = (1.0 - valid_score / y_valid.shape[0])

            print('\t', j, '\t', sep='', end=' ')
            print('\t|\t ', "{0:.5f}".format(self.train_loss[j]),
                  '  \t|\t ', "{0:.5f}".format(self.train_error[j]),
                  '  \t|\t ', "{0:.5f}".format(self.valid_loss[j]),
                  '  \t|\t ', "{0:.5f}".format(self.valid_error[j]),
                  '\t',
                  sep='')
        return


class NlpL3TypeB(nn.NeuralNetwork):
    def __init__(self):
        self.train_status = None
        self.train_settings = None
        self.layer0 = layer.Embedding(8000, 16)
        self.layer1_0 = layer.Linear(16, 128)
        self.layer1_1 = layer.Linear(16, 128)
        self.layer1_2 = layer.Linear(16, 128)
        self.layer2 = layer.Tanh(128)
        self.layer3 = layer.Linear(128, 8000)

        self.layer4 = layer.Softmax(8000)

        self.num_class = 8000

        self.learning_rate = None
        self.l1_regularize = None
        self.l2_regularize = None

        self.h0_0 = None
        self.h0_1 = None
        self.h0_2 = None

        self.h1_0 = None
        self.h1_1 = None
        self.h1_2 = None

        self.h2_0 = None
        self.h2_1 = None
        self.h2_2 = None
        self.h2 = None
        self.h3 = None
        self.h4 = None
        self.h5 = None

        # training parameters
        self.target_epoch = None
        self.current_epoch = None
        self.learning_rate = None
        self.l1_regularize = None
        self.l2_regularize = None
        self.batch_size = None
        self.momentum = None

        self.debug = None

        self.train_error = numpy.zeros(shape=(1,), dtype=numpy.float64)
        self.valid_error = numpy.zeros(shape=(1,), dtype=numpy.float64)
        self.train_loss = numpy.zeros(shape=(1,), dtype=numpy.float64)
        self.valid_loss = numpy.zeros(shape=(1,), dtype=numpy.float64)

    def fprop(self, x, keep_state: bool=False):
        if keep_state:
            self.h0_0 = x[:, 0]
            self.h0_1 = x[:, 1]
            self.h0_2 = x[:, 2]
            self.h1_0 = self.layer0.forward(self.h0_0)
            self.h1_1 = self.layer0.forward(self.h0_1)
            self.h1_2 = self.layer0.forward(self.h0_2)
            self.h2_0 = self.layer1_0.forward(self.h1_0)
            self.h2_1 = self.layer1_1.forward(self.h1_1)
            self.h2_2 = self.layer1_2.forward(self.h1_2)
            self.h2 = self.h2_0 + self.h2_1 + self.h2_2
            self.h3 = self.layer2.forward(self.h2)
            self.h4 = self.layer3.forward(self.h3)
            self.h5 = self.layer4.forward(self.h4)
            return self.h5
        else:
            h0_0 = x[:, 0]
            h0_1 = x[:, 1]
            h0_2 = x[:, 2]
            h1_0 = self.layer0.forward(h0_0)
            h1_1 = self.layer0.forward(h0_1)
            h1_2 = self.layer0.forward(h0_2)
            h2_0 = self.layer1_0.forward(h1_0)
            h2_1 = self.layer1_1.forward(h1_1)
            h2_2 = self.layer1_2.forward(h1_2)
            h2 = h2_0 + h2_1 + h2_2
            h3 = self.layer2.forward(h2)
            h4 = self.layer3.forward(h3)
            h5 = self.layer4.forward(h4)
            return h5

    def bprop(self, y):
        d_h4 = self.layer4.backward(y, self.h5, self.h4)
        d_h3 = self.layer3.backward(d_h4, self.h4, self.h3)
        d_h2 = self.layer2.backward(d_h3, self.h3, self.h2)
        # the gradiants of h2 is the same as h2_0, h2_1 and h2_2, which can be derived
        d_h1_2 = self.layer1_2.backward(d_h2, self.h2_2, self.h1_2)
        d_h1_1 = self.layer1_1.backward(d_h2, self.h2_1, self.h1_1)
        d_h1_0 = self.layer1_0.backward(d_h2, self.h2_0, self.h1_0)
        self.layer0.backward(d_h1_2, self.h1_2, self.h0_2)
        self.layer0.backward(d_h1_1, self.h1_1, self.h0_1)
        self.layer0.backward(d_h1_0, self.h1_0, self.h0_0)
        return

    def update(self, train_settings: uf.TrainSettings):
        self.layer0.update(train_settings.learning_rate)
        self.layer1_0.update(train_settings.learning_rate, momentum=train_settings.momentum)
        self.layer1_1.update(train_settings.learning_rate, momentum=train_settings.momentum)
        self.layer1_2.update(train_settings.learning_rate, momentum=train_settings.momentum)
        self.layer3.update(train_settings.learning_rate, momentum=train_settings.momentum)
        return


def loss_callback(status: uf.Status):
    current_epoch = status.current_epoch

    status.loss[current_epoch] += (uf.cross_entropy_loss(status.soft_prob,
                                                        status.y_batch,
                                                        take_average=False) / status.size)
    status.error[current_epoch] += (numpy.sum(status.predict != status.y_batch) / status.size)
    try:
        status.perplexity[current_epoch] += uf.perplexity(status.soft_prob) / status.size
    except TypeError:
        status.perplexity = numpy.zeros(shape=(status.target_epoch,), dtype=numpy.float32)
        tmp = uf.perplexity(status.soft_prob) / status.size
        status.perplexity[current_epoch] += tmp
    return


def plot_callback(status_train: uf.Status, status_valid: uf.Status):
    plotter.plot_loss(status_train, status_valid)
    plotter.plot_perplexity(status_train, status_valid)
    return





