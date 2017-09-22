import numpy
import pickle
from src import layer
import os


# initialize the neural network constructor before using
def init_nn(random_seed=1099):
    # set a random seed to ensure the consistence between different runs
    numpy.random.RandomState(seed=random_seed)
    return


class NeuralNetwork(object):
    def forward_propagation(self):
        # x shape (sample_dimension, 1)
        self._H[0] = self._x_batch
        for i in range(0, self._num_layer, 1):
            self._H[i + 1] = self._layers[i].forward(self._H[i])
        return

    def back_propagation(self):
        y = self._y_batch
        for i in range(self._num_layer-1, -1, -1):
            y = self._layers[i].backward(self._H[i + 1], self._H[i], y)
        return

    def update(self):
        for layer in self._layers:
            layer.update()

    def __init__(self, layers: [layer.Layer]):
        self._num_layer = len(layers)
        self._layers = layers
        self._H = [numpy.zeros((layers[0]._input_dimension, ), dtype=numpy.float64)]
        self._H += [[numpy.zeros((layer._output_dimension, ), dtype=numpy.float64)] for layer in self._layers]
        self._learning_rate = 0.01
        self._regularizer = 0.0001
        self._debug = False
        self._epoch = 20
        self._momentum = 0.9
        for layer in self._layers:
            layer.initialize()
        return

    def set_debug(self, debug):
        self._debug = debug

    def set_hyperparameter(self, learning_rate=0.01, regularizer=0.0001, momentum=0.9):
        self._learning_rate = learning_rate
        self._regularizer = regularizer
        self._momentum = momentum
        for layer in self._layers:
            layer.set_hyperparameter(learning_rate=learning_rate, momentum=momentum, regularizer=regularizer)

    def set_training(self, learning_rate=0.01, regularizer=0.0001, momentum=0.9, batch_size=128, epoch=200):
        self.set_hyperparameter(learning_rate, regularizer, momentum)
        self._batch_size = batch_size
        self._epoch = epoch
        self._H = [numpy.zeros((batch_size, self._layers[0]._input_dimension, 1), dtype=numpy.float64)]
        self._H += [[numpy.zeros((batch_size, layer._output_dimension, 1), dtype=numpy.float64)] for layer in self._layers]

    def feed_train(self, x, y):
        self._x_train = x
        self._y_train = y

    def shuffle(self):
        shuffle_idx = numpy.arange(self._y_train.shape[0])
        numpy.random.shuffle(shuffle_idx)
        self._x_shuffled = self._x_train[shuffle_idx]
        self._y_shuffled = self._y_train[shuffle_idx]

    def get_batch(self, i):
        self._x_batch = (self._x_shuffled[i:i + self._batch_size, :])
        self._y_batch = (self._y_shuffled[i:i + self._batch_size])
        offset = i + self._batch_size - self._y_train.shape[0]
        if offset > 0:
            a1 = (self._x_shuffled[0: offset, :])
            a2 = (self._y_shuffled[0: offset])
            self._x_batch = numpy.concatenate((self._x_batch, a1), axis=0)
            self._y_batch = numpy.concatenate((self._y_batch, a2))
        self._x_batch = self._x_batch.reshape(self._x_batch.shape[0], self._x_batch.shape[1], 1)

    def train(self):
        print('------------------ Start Training -----------------')
        print('\tepoch\t|\ttrain error\t|\ttrain loss\t')
        self._train_error = numpy.zeros(shape=(self._epoch,), dtype=numpy.float64)
        self._train_loss = numpy.zeros(shape=(self._epoch,), dtype=numpy.float64)

        for j in range(self._epoch):
            self.shuffle()
            for i in range(0, self._y_train.size, self._batch_size):
                self.get_batch(i)
                self.forward_propagation()
                self.back_propagation()
                self.update()
                self.get_train_loss(j)
            if (j + 1) % 20 == 0:
                self.dump(j)
            print('\t', j, '\t', sep='', end=' ')
            print('\t', self._train_error[j], '\t', '\t', self._train_loss[j], '\t', sep='')
            self.dump(j)

    def dump(self, epoch_num):
        full_path = os.path.realpath(__file__)
        path, _ = os.path.split(full_path)
        dump_file = os.path.join(path, '../temp', 'network-' + str(epoch_num) + '.dump')
        pickle.dump(self, open(dump_file, 'wb'))

    def get_train_loss(self, j):
        for i in range(self._batch_size):
            self._train_loss[j] += -numpy.log(self._H[-1][i, self._y_batch[i]]) / self._y_train.size

    def get_train_error(self, j):
        for i in range(self._batch_size):
            self._train_loss[j] += -numpy.log(self._H[-1][i, self._y_batch[i]]) / self._y_train.size

    def pick_class(self):
        max_prob = 0.0
        current_class = 0
        for j in range(self._num_class):
            current_prob = self._H[-1][j]
            if current_prob > max_prob:
                current_class = j
                max_prob = current_prob
        return current_class
