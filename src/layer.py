import numpy
import math

# initialize the neural network constructor before using
def init_nn(random_seed=1099):
    # set a random seed to ensure the consistence between different runs
    numpy.random.seed(random_seed)
    return

class Layer(object):
    # assume current layer is the k'th layer

    def __init__(self, input_dimension, output_dimension, beta=0.9):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        b = math.sqrt(6.0) / math.sqrt(input_dimension + output_dimension + 0.0)
        self.w = (numpy.random.rand(input_dimension, output_dimension) - 0.5) * 2 * b
        self.b = numpy.zeros((output_dimension, 1), dtype=numpy.float64)
        self.beta = beta
        self.delta_w = numpy.zeros(shape=self.w.shape, dtype=numpy.float64)
        self._delta_b = numpy.zeros(shape=self.b.shape, dtype=numpy.float64)

    def activation(self, a):
        raise ValueError('Calling a virtual function')

    def derivative(self, h):
        raise ValueError('Calling a virtual function')

    def forward(self, x):
        return self.activation(numpy.dot(self.w.transpose(), x) + self.b)

    def gradient_a(self, gradient_h, h):
        g_a = numpy.zeros((1, self.output_dimension), dtype=numpy.float64)
        deri = self.derivative(h)
        deri.shape[0]
        for i in range(deri.shape[0]):
            g_a[0, i] = deri[i] * gradient_h[0, i]
        return g_a

    def update_w(self, learning_rate, regular, g_w):
        self.delta_w = g_w.transpose() + self.beta * self.delta_w
        delta = -self.delta_w - regular * 2.0 * self.w
        self.w += learning_rate * delta

    def update_b(self, learning_rate, regular, g_b):
        self._delta_b = g_b.transpose() + self.beta * self._delta_b
        delta = -self._delta_b
        self.b += learning_rate * delta


class SigmoidLayer(Layer):
    def __init__(self, input_dimension, output_dimension, prev_layer=None):
        Layer.__init__(self, input_dimension, output_dimension)

    def activation(self, x):
        x = numpy.clip(x, -500.0, 500.0)
        return numpy.array([1.0 / (1.0 + math.exp(-xi)) for xi in x]).reshape(x.shape)

    def derivative(self, h):
        return numpy.array([hi * (1 - hi) for hi in h])



class SoftmaxLayer(Layer):
    def __init__(self, input_dimension, output_dimension):
        Layer.__init__(self, input_dimension, output_dimension)

    def activation(self, x):
        expi = numpy.array([math.exp(xi) for xi in x])
        return (expi / numpy.sum(expi)).reshape(x.shape[0], 1)


    def derivative(self, h):
        raise ValueError('''Softmax Layer doesn't need derivative''')



class NeuralNetwork(object):

    @property
    def num_class(self):
        return self._num_class

    def __init__(self, layer_list: [Layer], learning_rate=0.01, regularizer=0.00, num_class=10, debug=False):
        self.num_layer = len(layer_list)
        self.layers = layer_list
        self._H = [numpy.zeros((layer_list[0].input_dimension, 1), dtype=numpy.float64)]
        self._H += [[numpy.zeros((layer.output_dimension, 1), dtype=numpy.float64)] for layer in self.layers]
        self._alpha = learning_rate
        self._lambda = regularizer
        self._num_class = num_class
        self._debug = debug
        self.epoch = 200


    def forward_propagation(self, x):
        # x shape (sample_dimension, 1)
        self._H[0] = x.reshape(x.shape[0], 1)
        for i in range(0, self.num_layer, 1):
            self._H[i + 1] = self.layers[i].forward(self._H[i])
        return self._H[self.num_layer]


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
        g_a = self._H[self.num_layer].transpose()
        # Assuming label y range from 0, 1, ...,
        g_a[0, y] -= 1.0
        for k in range(self.num_layer, 1, -1):
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
        for k in range(self.num_layer, 1, -1):
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

    def train_setting(self, epoch=100, learning_rate = 0.01, batch_size = 128, weight_decay=0.001, momentum= 0.9, early_stop=False, debug=False):
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
                    for k in range(self.num_layer):
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
            for j in range(self.num_class):
                current_prob = self._H[-1][j]
                if current_prob > max_prob:
                    current_class = j
                    max_prob = current_prob
            Y[i] = current_class
        return Y

    def pick_class(self):
        max_prob = 0.0
        current_class = 0
        for j in range(self.num_class):
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


class SingleLayerNetwork(NeuralNetwork):
    def __init__(self, input_dimension, hidden_dimension, output_dimension, learning_rate=0.01, debug=False):
        layer0 = SigmoidLayer(input_dimension, hidden_dimension)
        layer1 = SoftmaxLayer(hidden_dimension, output_dimension)
        layer_list = [layer0, layer1]
        NeuralNetwork.__init__(self, layer_list, learning_rate=learning_rate, debug=debug)


class MultiLayerNetwork(NeuralNetwork):
    def __init__(self, input_dimension, output_dimension, learning_rate=0.01, debug=False):
        layer0 = SigmoidLayer(input_dimension, 100)
        layer1 = SigmoidLayer(100, 50)
        layer2 = SigmoidLayer(50, 25)
        layer3 = SoftmaxLayer(25, output_dimension)
        layer_list = [layer0, layer1, layer2, layer3]
        NeuralNetwork.__init__(self, layer_list, learning_rate=learning_rate, debug=debug)