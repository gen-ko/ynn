import numpy
import pickle
from src import layer
import os
from time import gmtime, strftime
import matplotlib.pyplot as plt



# initialize the neural network constructor before using
def init_nn(random_seed=1099):
    # set a random seed to ensure the consistence between different runs
    numpy.random.RandomState(seed=random_seed)
    return




class NeuralNetwork_Dumpable(object):
    def __init__(self, layers: [layer.FullConnectLayer], learning_rate=0.01, regularizer=0.0001, debug=False,
                 momentum=0.9):

        self._num_layer = len(layers)
        self.layers = layers
        self._learning_rate = learning_rate
        self._regularizer = regularizer
        self._H = [numpy.zeros((layers[0]._input_dimension, ), dtype=numpy.float64)]
        self._H += [[numpy.zeros((layer._output_dimension, ), dtype=numpy.float64)] for layer in self.layers]
        self._num_class = layers[-1]._output_dimension
        self._debug = debug
        self.epoch = 200
        self._momentum = momentum
        self.g_h: list
        self.train_error = numpy.zeros(shape=(1,), dtype=numpy.float64)
        self.valid_error = numpy.zeros(shape=(1,), dtype=numpy.float64)
        self.train_loss = numpy.zeros(shape=(1,), dtype=numpy.float64)
        self.valid_loss = numpy.zeros(shape=(1,), dtype=numpy.float64)

    def forward_propagation(self, x):
        # x shape (sample_dimension, 1)
        self._H[0] = x.reshape(x.shape[0], 1)
        for i in range(0, self._num_layer, 1):
            self._H[i + 1] = self.layers[i].forward(self._H[i])
        return self._H[self._num_layer]

    def fprop(self, x):
        self._H[0] = x
        for i in range(0, self._num_layer, 1):
            self._H[i + 1] = self.layers[i].forward(self._H[i])

    def bprop(self, y):
        g_h = y
        for i in range(self._num_layer - 1, -1, -1):
            g_h = self.layers[i].backward(g_h, h_out=self._H[i + 1], h_in=self._H[i])

    def update(self):
        for layer in self.layers:
            layer.update(self._learning_rate, self._regularizer, self._momentum)

    def shuffle(self, x, y):
        shuffle_idx = numpy.arange(y.size)
        numpy.random.shuffle(shuffle_idx)
        x = x[shuffle_idx]
        y = y[shuffle_idx]
        return x, y



    def train(self, x_train, y_train, x_valid, y_valid, epoch, batch_size=128, dump=False):
        self._dump = dump
        print('------------------ Start Training -----------------')
        print('\tepoch\t|\ttrain loss\t|\ttrain error\t|\tvalid loss\t|\tvalid error\t')
        self.train_error = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        self.valid_error = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        self.train_loss = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        self.valid_loss = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        self.batch_size = batch_size
        for j in range(epoch):
            x_train, y_train = self.shuffle(x_train, y_train)
            train_score = 0.0
            for i in range(0, y_train.size, batch_size):
                x_batch = x_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]
                x_batch = x_batch.T

                self.fprop(x_batch)
                self.train_loss[j] += self.cross_entropy_loss(y_batch)
                y_predict = self.pick_class()
                train_score += numpy.sum(y_predict == y_batch)
                self.bprop(y_batch)
                self.update()
            # start a validation
            self.train_loss[j] = self.train_loss[j] / y_train.shape[0]
            self.train_error[j] = (1.0 - train_score / y_train.shape[0])
            self.valid_pass(x_valid, y_valid, j)


            print('\t', j, '\t', sep='', end=' ')
            print('\t|\t ', "{0:.5f}".format(self.train_loss[j]),
                  '  \t|\t ', "{0:.5f}".format(self.train_error[j]),
                  '  \t|\t ', "{0:.5f}".format(self.valid_loss[j]),
                  '  \t|\t ', "{0:.5f}".format(self.valid_error[j]),
                  '\t',
                  sep='')
            if (j + 1) % 20 == 0:
                self.dump()
                self.visualize(j)
        self.plot()
        return


    def valid_pass(self, x_valid, y_valid, epoch):
        self.fprop(x_valid.T)
        self.valid_loss[epoch] = self.cross_entropy_loss(y_valid)
        y_predict = self.pick_class()
        valid_score = numpy.sum(y_predict == y_valid)
        self.valid_loss[epoch] /= y_valid.shape[0]
        self.valid_error[epoch] = 1.0 - valid_score / y_valid.shape[0]

    def dump(self):
        if not self._dump:
            return
        # fix the absolute path of file
        full_path = os.path.realpath(__file__)
        path, _ = os.path.split(full_path)
        time = strftime("%y%m%d%H%M%S", gmtime())
        filename = time + '-network.dump'
        figure_path = os.path.join(path, '../temp', filename)
        pickle.dump(self, open(figure_path, 'wb'))


    def cross_entropy_loss(self, y):
        loss = 0.0
        for i in range(y.size):
            h_out = self._H[-1]
            try:
                loss += -numpy.log(h_out[y[i], i])
            except:
                print('er')
        return loss
        

    def predict(self, X):
        Y = numpy.zeros((X.shape[0],))
        for i in range(X.shape[0]):
            x = X[i]
            x = x.reshape(x.shape[0], 1)
            self.forward_propagation(x)
            max_prob = 0.0
            current_class = 0
            for j in range(self._num_class):
                current_prob = self._H[-1][j]
                if current_prob > max_prob:
                    current_class = j
                    max_prob = current_prob
            Y[i] = current_class
        return Y

    def pick_class(self):

        current_class = numpy.zeros(len(self._H[-1][0, :]))
        for i in range(current_class.size):
            max_prob = 0.0
            for j in range(self._num_class):
                current_prob = self._H[-1][j, i]
                if current_prob > max_prob:
                    current_class[i] = j
                    max_prob = current_prob
        return current_class

    def score(self, X, Y):
        Y_predicted = self.predict(X)
        correct_count = 0
        for i in range(Y.size):
            if Y[i] == Y_predicted[i]:
                correct_count += 1
        return (correct_count + 0.0) / float(Y.size)

    def debug_gradient_w(self, k, y):
        self.debug_gradient_w()
        # g_w_debug = g_w wrong! this is copy by reference
        w_shape = self.layers[k - 1].w.shape
        g_w_debug = numpy.zeros((w_shape[1], w_shape[0]), dtype=numpy.float64)
        for i in range(w_shape[0]):
            for j in range(w_shape[1]):
                self.layers[k - 1].w[i, j] += 0.001
                self.forward_propagation(self._H[0])
                Hl = -numpy.log(self._H[-1][y])
                self.layers[k - 1].w[i, j] -= 0.002
                self.forward_propagation(self._H[0])
                Hr = -numpy.log(self._H[-1][y])
                g_w_debug[j, i] = (Hl - Hr) / 0.002
                self.layers[k - 1].w[i, j] += 0.001

        self.forward_propagation(self._H[0])

    def debug_gradient_b(self, k, y):
        b_shape = self.layers[k - 1].b.shape
        g_b_debug = numpy.zeros((b_shape[1], b_shape[0]), dtype=numpy.float64)
        for i in range(b_shape[0]):
            for j in range(b_shape[1]):
                self.layers[k - 1].b[i, j] += 0.001
                self.forward_propagation(self._H[0])
                Hl = -numpy.log(self._H[-1][y])
                self.layers[k - 1].b[i, j] -= 0.002
                self.forward_propagation(self._H[0])
                Hr = -numpy.log(self._H[-1][y])
                g_b_debug[j, i] = (Hl - Hr) / 0.002
                self.layers[k - 1].b[i, j] += 0.001
        self.forward_propagation(self._H[0])

    def visualize(self, epoch):
        # fix the absolute path of file
        full_path = os.path.realpath(__file__)
        path, _ = os.path.split(full_path)
        time = strftime("%y%m%d%H%M%S", gmtime())
        filename = time + '-weights.png'
        figure_path = os.path.join(path, '../output/visualize', filename)

        ncol = 10
        nrow = int(self.layers[0].w.shape[1] / ncol)

        fig = plt.figure()
        plt.axis('off')
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        for i in range(self.layers[0].w.shape[1]):
            ax = plt.subplot(nrow, ncol, i+1)
            im = ax.imshow(self.layers[0].w[:, i].reshape([28, 28]))
            ax.axis('off')
        plt.savefig(os.path.join(path, figure_path))
        return

    def plot(self):
        full_path = os.path.realpath(__file__)
        path, _ = os.path.split(full_path)
        time = strftime("%y%m%d%H%M%S", gmtime())
        filename = time + '-loss.png'
        titletext = f'lr={self._learning_rate},l2={self._regularizer},m={self._momentum},b={self.batch_size}'

        plt.figure()
        line_1, = plt.plot(self.train_loss, label='train loss')
        line_2, = plt.plot(self.valid_loss, label='valid loss')
        plt.legend(handles=[line_1, line_2])
        plt.xlabel('epoch')
        plt.ylabel('cross-entropy loss')
        plt.title(titletext)
        plt.savefig(os.path.join(path, '../output/plot-loss', filename))

        plt.figure()
        line_1, = plt.plot(self.train_error, label='train error')
        line_2, = plt.plot(self.valid_error, label='valid error')
        plt.legend(handles=[line_1, line_2])
        plt.xlabel('epoch')
        plt.ylabel('error rate')
        plt.title(titletext)
        plt.savefig(os.path.join(path, '../output/plot-error', filename))
        return
