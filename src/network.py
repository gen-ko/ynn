import numpy
import pickle
from src import layer


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

    def train(self, x_train, y_train, x_valid, y_valid, epoch, dump_file, batch_size=1):
        print('------------------ Start Training -----------------')
        print('\tepoch\t|\ttrain error\t|\tvalid error\t|\ttrain loss\t|\tvalid loss\t')
        self._train_error = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        self._valid_error = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        self._train_loss = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        self._valid_loss = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        for j in range(epoch):
            shuffle_idx = numpy.arange(y_train.shape[0])
            numpy.random.shuffle(shuffle_idx)
            train_score = 0.0
            for i in range(0, y_train.size, batch_size):
                idx = shuffle_idx[i:i+batch_size]
                x = x_train[idx]
                y_batch = y_train[idx]
                x = x.T
                self.fprop(x)
                self._train_loss[j] += self.cross_entropy_loss(y_batch)
                y = self.pick_class()
                if y == y_train[idx]:
                    train_score += 1.0
                self.bprop(y_batch)
            # start a validation
            self._train_loss[j] = self._train_loss[j] / y_train.shape[0]
            self._train_error[j] = (1.0 - train_score / y_train.shape[0])

            valid_score = 0.0
            for i in range(y_valid.size):
                self.forward_propagation(x_valid[i].reshape(x_valid[i].shape[0], 1))
                self._valid_loss[j] += self.cross_entropy_loss(y_valid[i])
                y = self.pick_class()
                if y == y_valid[i]:
                    valid_score += 1.0

            self._valid_loss[j] = self._valid_loss[j] / y_train.shape[0]
            self._valid_error[j] = (1.0 - valid_score / y_valid.shape[0])

            print('\t', j, '\t', sep='', end=' ')
            print('\t|\t ', "{0:.5f}".format(self._train_error[j]),
                  '  \t|\t ', "{0:.5f}".format(self._valid_error[j]),
                  '  \t|\t ', "{0:.5f}".format(self._train_loss[j]),
                  '  \t|\t ', "{0:.5f}".format(self._valid_loss[j]), '\t',
                  sep='')
        self.dump(dump_file=dump_file)




    def dump(self, dump_file='./temp/network.dump'):
        pickle.dump(self, open(dump_file, 'wb'))


    def cross_entropy_loss(self, y):
        loss = 0.0
        for i in range(len(y)):
            loss += -numpy.log(self._H[-1][y[i], i])
        return loss
        

    def train_setting(self, epoch=100, learning_rate = 0.01, batch_size = 128, weight_decay=0.00, momentum= 0.0, early_stop=False, debug=False):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self._momentum = momentum
        self.batch_size = batch_size
        self.early_stop = early_stop
        self._debug = debug

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
        max_prob = 0.0
        current_class = numpy.zeros(len(self._H[-1][0, :]))
        for i in range(current_class.size):
            for j in range(self._num_class):
                current_prob = self._H[-1][j]
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
