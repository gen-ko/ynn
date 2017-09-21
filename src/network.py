import numpy
import pickle
from src import layer

# initialize the neural network constructor before using
def init_nn(random_seed=1099):
    # set a random seed to ensure the consistence between different runs
    nn_random_state = numpy.random.RandomState(seed=random_seed)
    return

class NeuralNetwork(object):
    def __init__(self, layer_list: [layer.Layer], num_class, learning_rate=0.01, regularizer=0.0001, debug=False):
        self._num_layer = len(layer_list)
        self.layers = layer_list
        self._H = [numpy.zeros((layer_list[0].input_dimension, 1), dtype=numpy.float64)]
        self._H += [[numpy.zeros((layer.output_dimension, 1), dtype=numpy.float64)] for layer in self.layers]
        self._alpha = learning_rate
        self._lambda = regularizer
        self._num_class = num_class
        self._debug = debug
        self.epoch = 200
        
    def update(self):
        for i in range(0, self._num_layer, 1):
            self.layers[i].update(self._delta_w[i], self._delta_b[i])


    def forward_propagation(self, x):
        # x shape (sample_dimension, 1)
        self._H[0] = x.reshape(x.shape[0], 1)
        for i in range(0, self._num_layer, 1):
            self._H[i + 1] = self.layers[i].forward_slow(self._H[i])
        return self._H[self._num_layer]


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

    def back_propagation(self, y):
        g_a = self._H[self._num_layer].transpose()
        # Assuming label y range from 0, 1, ...,
        g_a[0, y] -= 1.0
        for k in range(self._num_layer, 1, -1):
            g_w = numpy.dot(g_a.transpose(), self._H[k - 1].transpose())
            if self._debug:
                self.debug_gradient_w(k, y)
                self.debug_gradient_b(k, y)
            self.layers[k - 1].update_w(self._alpha, self._lambda, g_w)
            self.layers[k - 1].update_b(self._alpha, self._lambda, g_a)
            g_h = numpy.dot(g_a, self.layers[k - 1].w.transpose())
            g_a = self.layers[k - 2].gradient_a(g_h, self._H[k - 1])

        # for the input layer
        g_w = numpy.dot(g_a.transpose(), self._H[0].transpose())
        self.layers[0].update_w(self._alpha, self._lambda, g_w)
        g_b = g_a
        self.layers[0].update_b(self._alpha, self._lambda, g_b)

    def back_propagation_batch(self, avg_H):
        g_a = avg_H[-1].transpose()
        for k in range(self._num_layer, 1, -1):
            g_w = numpy.dot(g_a.transpose(), avg_H[k - 1].transpose())
            self.layers[k - 1].update_w(self._alpha, self._lambda, g_w)
            g_b = g_a
            self.layers[k - 1].update_b(self._alpha, self._lambda, g_b)
            g_h = numpy.dot(g_a, self.layers[k - 1].w.transpose())
            g_a = self.layers[k - 2].gradient_a(g_h, avg_H[k - 2])

        # for the input layer
        g_w = numpy.dot(g_a.transpose(), avg_H[0].transpose())
        self.layers[0].update_w(self._alpha, self._lambda, g_w)
        self.layers[0].update_b(self._alpha, self._lambda, g_a)

    def train(self, x_train, y_train, x_valid, y_valid, epoch):
        print('------------------ Start Training -----------------')
        print('\tepoch\t|\ttrain score\t|\tvalid score\t|\t')
        for j in range(epoch):
            print('\t', j, '\t', sep='', end=' ')
            shuffle_idx = numpy.arange(y_train.shape[0])
            numpy.random.shuffle(shuffle_idx)
            train_score = 0.0
            for i in range(y_train.size):
                idx = shuffle_idx[i]
                x = x_train[idx]
                x = x.reshape(x.shape[0], 1)
                self.forward_propagation(x)
                y = self.pick_class()
                if y == y_train[idx]:
                    train_score += 1.0

                self.back_propagation(y_train[idx])
            # start a validation
            print('\t', train_score/y_train.shape[0], '\t', self.score(x_valid, y_valid), '\t', sep='')

    def train_trace(self, x_train, y_train, x_valid, y_valid, epoch):
        print('------------------ Start Training -----------------')
        print('\tepoch\t|\ttrain error\t|\tvalid error\t|\t')
        train_error = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        valid_error = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        for j in range(epoch):
            
            shuffle_idx = numpy.arange(y_train.shape[0])
            numpy.random.shuffle(shuffle_idx)
            train_score = 0.0
            for i in range(y_train.size):
                idx = shuffle_idx[i]
                x = x_train[idx]
                x = x.reshape(x.shape[0], 1)
                self.forward_propagation(x)
                y = self.pick_class()
                if y == y_train[idx]:
                    train_score += 1.0

                self.back_propagation(y_train[idx])
            # start a validation

            train_error[j] = (1.0 - train_score/y_train.shape[0])
            valid_error[j] = (1.0 - self.score(x_valid, y_valid))
            train_error.dump('./temp/train_error.dump')
            valid_error.dump('./temp/valid_error.dump')
            print('\t', j, '\t', sep='', end=' ')
            print('\t', train_error[j], '\t', valid_error[j], '\t', sep='')
        self.dump()
    
    def dump(self):
        pickle.dump(self, open('./temp/network.dump', 'wb'))
        

    def train_setting(self, epoch=100, learning_rate = 0.01, batch_size = 128, weight_decay=0.00, momentum= 0.0, early_stop=False, debug=False):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.early_stop = early_stop
        self._debug = debug

    def train_in_batch(self, x_train, y_train, x_valid, y_valid):
        print('------------------ Start Training -----------------')
        print('\tepoch\t|\ttrain score\t|\tvalid score\t')
        for j in range(self.epoch):
            print('\t', j, '\t', sep='', end=' ')
            shuffle_idx = numpy.arange(y_train.shape[0])
            numpy.random.shuffle(shuffle_idx)
            train_score = 0.0

            for i in range(0, y_train.shape[0], self.batch_size):
                idx_batch = shuffle_idx[i: i + self.batch_size]
                x_batch = x_train[idx_batch]
                sum_h = [numpy.zeros((self.layers[0].input_dimension, 1))]
                sum_h += [numpy.zeros((layer.output_dimension, 1), dtype=numpy.float64) for layer in self.layers]
                for t in range(0, x_batch.shape[0], 1):
                    x = x_batch[t]
                    x = x.reshape(x.shape[0], 1)
                    self.forward_propagation(x)
                    #idx = idx_batch[t]
                    #yi = y_train[idx]
                    #h = sum_h[-1]
                    #h[yi] -= 1.0
                    sum_h[-1][y_train[idx_batch[t]]] -= 1.0
                    y = self.pick_class()
                    if y == y_train[idx_batch[t]]:
                        train_score += 1.0
                    for k in range(self._num_layer):
                        sum_h[k] += self._H[k] / x_batch.shape[0]
                self.back_propagation_batch(sum_h)

            print('\t', train_score / y_train.shape[0], '\t', self.score(x_valid, y_valid), '\t', sep='')

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
        current_class = 0
        for j in range(self._num_class):
            current_prob = self._H[-1][j]
            if current_prob > max_prob:
                current_class = j
                max_prob = current_prob
        return current_class

    def score(self, X, Y):
        Y_predicted = self.predict(X)
        correct_count = 0
        for i in range(Y.size):
            if Y[i] == Y_predicted[i]:
                correct_count += 1
        return (correct_count + 0.0) / float(Y.size)





class MultiLayerNetwork(NeuralNetwork):
    def __init__(self, input_dimension, output_dimension, learning_rate=0.01, debug=False):
        layer0 = layer.SigmoidLayer(input_dimension, 100)
        layer1 = layer.SigmoidLayer(100, 50)
        layer2 = layer.SigmoidLayer(50, 25)
        layer3 = layer.SoftmaxLayer(25, output_dimension)
        layer_list = [layer0, layer1, layer2, layer3]
        NeuralNetwork.__init__(self, layer_list, learning_rate=learning_rate, debug=debug)