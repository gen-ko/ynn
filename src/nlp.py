import numpy
import pickle
from src import layer
import os
from time import gmtime, strftime
import matplotlib.pyplot as plt
from src import util as uf
from src import network as nn
from src import plotter

name_pool = layer.name_pool

class NlpGeneral(nn.NeuralNetwork):
    def __init__(self, layers: [layer.Layer], connections: dict):
        nn.NeuralNetwork.__init__(self, layers, connections)
        self.next_layers: dict = {}
        self.prev_layers: dict = {}
        self.input_layers: list = []
        self.h_out: dict = {}
        self.h_in: dict = {}
        self.layer_order: list = []
        self.output: numpy.ndarray
        for key in connections:
            if key is 'input':
                layer_list: list = []
                for name in connections[key]:
                    layer_list += name_pool[name]
                self.input_layers = layer_list
            elif key is 'output':
                self.output_layer = connections[key][0]
            else:
                current_layer = name_pool[key]
                self.layer_order += current_layer
                for name in connections[key]:
                    next_layer = name_pool[name]
                    try:
                        self.next_layers[current_layer] += next_layer
                    except KeyError:
                        self.next_layers[current_layer] = [next_layer]
                    try:
                        self.prev_layers[next_layer] += current_layer
                    except KeyError:
                        self.prev_layers[next_layer] = [current_layer]
        return


    def fprop(self, x, keep_state: bool=False):
        input_layer_size = len(self.input_layers)
        h_in: dict = {}
        h_out: dict = {}
        for i in range(input_layer_size):
            current_layer: layer.Layer = self.input_layers[i]
            h_in[current_layer] = x[:, i]

        for i in range(len(self.layer_order)):
            current_layer: layer.Layer = self.layer_order[i]
            if len(self.prev_layers[current_layer]) > 1:
                for i in range(len(self.prev_layers[current_layer])):
                    prev_layer = self.prev_layers[current_layer][i]
                    try:
                        h_in[current_layer] += h_out[prev_layer]
                    except NameError:
                        h_in[current_layer] = h_out[prev_layer]
            h_out[current_layer] = current_layer.forward(h_in[current_layer])
            h_in[self.next_layers[current_layer]] = h_out[current_layer]

        h_out[self.output_layer] = self.output_layer.forward(h_out[self.output_layer])
        if keep_state:
            self.h_in = h_in
            self.h_out = h_out

        return h_out[self.output_layer]


    def bprop(self, y):
        d_h_top: dict = {}
        d_h_down: dict = {}

        d_h_down[self.output_layer] = self.output_layer.backward(y,
                                                                 self.h_out[self.output_layer],
                                                                 self.h_in[self.output_layer])





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
    def __init__(self, dict_size: int, embedding_size: int, hidden_units: int):
        self.layers = [layer.Embedding(dict_size, embedding_size, 'l0'),     # layer 0
                       layer.Linear(embedding_size, hidden_units, 'l1-0'),   # layer 1
                       layer.Linear(embedding_size, hidden_units, 'l1-1'),   # layer 2
                       layer.Linear(embedding_size, hidden_units, 'l1-2'),   # layer 3
                       layer.Tanh(hidden_units, 'l2'),                       # layer 4
                       layer.Linear(hidden_units, dict_size, 'l3'),          # layer 5
                       layer.Softmax(dict_size, 'l4')]                       # layer 6
        self.h: list = None
        return

    def fprop(self, x, keep_state: bool=False):
        h = list()
        h.append(x[:, 0])  # h0
        h.append(x[:, 1])  # h1
        h.append(x[:, 2])  # h2
        h.append(self.layers[0].forward(h[0]))  # h3
        h.append(self.layers[0].forward(h[1]))  # h4
        h.append(self.layers[0].forward(h[2]))  # h5
        h.append(self.layers[1].forward(h[3]))  # h6
        h.append(self.layers[2].forward(h[4]))  # h7
        h.append(self.layers[3].forward(h[5]))  # h8
        h.append(h[6] + h[7] + h[8])            # h9
        h.append(self.layers[4].forward(h[9]))  # h10
        h.append(self.layers[5].forward(h[10]))  # h11
        h.append(self.layers[6].forward(h[11]))  # h12

        if keep_state:
            self.h = h

        return h[-1]

    def bprop(self, y):
        d_h4 = self.layers[6].backward(y, self.h[12], self.h[11])
        d_h3 = self.layers[5].backward(d_h4, self.h[11], self.h[10])
        d_h2 = self.layers[4].backward(d_h3, self.h[10], self.h[9])
        # the gradiants of h2 is the same as h2_0, h2_1 and h2_2, which can be derived
        d_h1_2 = self.layers[3].backward(d_h2, self.h[8], self.h[5])
        d_h1_1 = self.layers[2].backward(d_h2, self.h[7], self.h[4])
        d_h1_0 = self.layers[1].backward(d_h2, self.h[6], self.h[3])
        self.layers[0].backward(d_h1_2, self.h[5], self.h[2])
        self.layers[0].backward(d_h1_1, self.h[4], self.h[1])
        self.layers[0].backward(d_h1_0, self.h[3], self.h[0])
        return

class NlpL3TypeA(nn.NeuralNetwork):
    def __init__(self, dict_size: int, embedding_size: int, hidden_units: int):
        self.layers = [layer.Embedding(dict_size, embedding_size, 'l0'),
                       layer.Linear(embedding_size, hidden_units, 'l1-0'),
                       layer.Linear(embedding_size, hidden_units, 'l1-1'),
                       layer.Linear(embedding_size, hidden_units, 'l1-2'),
                       layer.Linear(hidden_units, dict_size, 'l2'),
                       layer.Softmax(dict_size, 'l3')]
        self.h: list = None
        return

    def fprop(self, x, keep_state: bool=False):
        h = list()
        h.append(x[:, 0])  # h0
        h.append(x[:, 1])  # h1
        h.append(x[:, 2])  # h2
        h.append(self.layers[0].forward(h[0]))  # h3
        h.append(self.layers[0].forward(h[1]))  # h4
        h.append(self.layers[0].forward(h[2]))  # h5
        h.append(self.layers[1].forward(h[3]))  # h6
        h.append(self.layers[2].forward(h[4]))  # h7
        h.append(self.layers[3].forward(h[5]))  # h8
        h.append(h[6] + h[7] + h[8])            # h9
        h.append(self.layers[4].forward(h[9]))  # h10
        h.append(self.layers[5].forward(h[10]))  # h11

        if keep_state:
            self.h = h

        return h[-1]

    def bprop(self, y):
        d_h3 = self.layers[5].backward(y, self.h[11], self.h[10])
        d_h2 = self.layers[4].backward(d_h3, self.h[10], self.h[9])
        # the gradiants of h2 is the same as h2_0, h2_1 and h2_2, which can be derived
        d_h1_2 = self.layers[3].backward(d_h2, self.h[8], self.h[5])
        d_h1_1 = self.layers[2].backward(d_h2, self.h[7], self.h[4])
        d_h1_0 = self.layers[1].backward(d_h2, self.h[6], self.h[3])
        self.layers[0].backward(d_h1_2, self.h[5], self.h[2])
        self.layers[0].backward(d_h1_1, self.h[4], self.h[1])
        self.layers[0].backward(d_h1_0, self.h[3], self.h[0])
        return


class NlpL3TypeB2(nn.NeuralNetwork):
    def __init__(self):
        self.train_status = None
        self.train_settings = None
        self.layer0 = layer.Embedding(8000, 16, 'l0')
        self.layer1_0 = layer.Linear(16, 128, 'l1-0')
        self.layer1_1 = layer.Linear(16, 128, 'l1-1')
        self.layer1_2 = layer.Linear(16, 128, 'l1-2')
        self.layer2 = layer.Tanh(128, 'l2')
        self.layer3 = layer.Linear(128, 8000, 'l3')

        self.layer4 = layer.Softmax(8000, 'l4')

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






