import numpy
import math

# initialize the neural network constructor before using
def init_nn(random_seed=1099):
    # set a random seed to ensure the consistence between different runs
    numpy.random.seed(random_seed)
    return

class Layer(object):
    # assume current layer is the k'th layer
    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b

    @property
    def input_dimension(self):
        return self._input_dimension

    @property
    def output_dimension(self):
        return self._output_dimension

    def __init__(self, input_dimension, output_dimension):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._w = (numpy.random.rand(input_dimension, output_dimension) - 0.5) / 10.0
        self._b = (numpy.random.rand(output_dimension, 1) - 0.5) / 10.0

    def activation(self, a):
        raise ValueError('Calling a virtual function')

    def derivative(self, h):
        raise ValueError('Calling a virtual function')

    def forward(self, x):
        return self.activation(numpy.dot(self._w.transpose(), x) + self._b)

    def gradient_w(self, gradient_a, h):
        return numpy.dot(gradient_a.transpose(), h.transpose())

    def gradient_b(self, gradient_a):
        return self.gradient_a

    def gradient_h(self, gradient_a):
        # gradient_a dimension (1, output_dimension)
        # w dimension (input_dimension, output_dimension)
        # gradient_w = w * gradient_a
        # gradient_h dimension (1, input_dimension)
        return numpy.dot(gradient_a, self._w.transpose())

    def gradient_a(self, gradient_h, h):
        g_a = numpy.zeros((1, self._output_dimension), dtype=numpy.float64)
        deri = self.derivative(h)
        for di in deri:
            g_a += gradient_h * di
        return g_a

    def update_w(self, learning_rate, regular, g_w):
        delta = -g_w.transpose() - regular * 2.0 * self._w
        self._w += learning_rate * delta

    def update_b(self, learning_rate, regular, g_b):
        delta = -g_b.transpose()
        self._b += learning_rate * delta


class SigmoidLayer(Layer):
    def __init__(self, input_dimension, output_dimension, prev_layer=None):
        Layer.__init__(self, input_dimension, output_dimension)

    def activation(self, x):
        return numpy.array([1.0 / (1.0 + math.exp(-xi)) for xi in x]).reshape(self.output_dimension, 1)

    def derivative(self, h):
        res = h
        for i in range(h.size):
            hi = h[i]
            res[i] = hi * (1 - hi)
        return res


class SoftmaxLayer(Layer):
    def __init__(self, input_dimension, output_dimension):
        Layer.__init__(self, input_dimension, output_dimension)

    def activation(self, x):
        expi = numpy.array([math.exp(xi) for xi in x]).reshape(self.output_dimension, 1)
        return expi / numpy.sum(expi)

    def derivative(self, h):
        raise ValueError('''Softmax Layer doesn't need derivative''')



class NeuralNetwork(object):
    @property
    def num_layer(self):
        return self._num_layer

    @property
    def layers(self):
        return self._layers

    @property
    def num_class(self):
        return self._num_class

    def __init__(self, layer_list: [Layer], learning_rate=0.0001, regularizer=0.0, num_class=10, debug=False):
        self._num_layer = len(layer_list)
        self._layers = layer_list
        self._H = [numpy.zeros(layer_list[0].input_dimension)]
        self._H += [[numpy.zeros((layer.output_dimension, 1), dtype=numpy.float64)] for layer in self._layers]
        self._alpha = learning_rate
        self._lambda = regularizer
        self._num_class = num_class
        self._debug = debug

    def forward_propagation(self, x):
        # x shape (sample_dimension, 1)
        self._H[0] = x
        for i in range(1, self._num_layer + 1):
            self._H[i] = self._layers[i - 1].forward(self._H[i - 1])
        return self._H[self.num_layer]

    def back_propagation(self, y):
        g_a = self._H[self.num_layer].transpose()
        # Assuming label y range from 0, 1, ...,
        g_a[0, y] -= 1.0
        for k in range(self._num_layer, 1, -1):
            g_w = numpy.dot(g_a.transpose(), self._H[k - 1].transpose())
            if self._debug:
                # g_w_debug = g_w wrong! this is copy by reference
                w_shape = self._layers[k - 1]._w.shape
                g_w_debug = numpy.zeros((w_shape[1], w_shape[0]), dtype=numpy.float64)
                for i in range(w_shape[0]):
                    for j in range(w_shape[1]):
                        self._layers[k - 1]._w[i, j] += 0.001
                        self.forward_propagation(self._H[0])
                        Hl =  -numpy.log(self._H[-1][y])

                        self._layers[k - 1]._w[i, j] -= 0.002
                        self.forward_propagation(self._H[0])
                        Hr =  -numpy.log(self._H[-1][y])
                        g_w_debug[j, i] = (Hl - Hr) / 0.002
                        self._layers[k - 1]._w[i, j] += 0.001
                self.forward_propagation(self._H[0])

            #g_w = self._layers[k - 1].gradient_w(g_a, self._H[k - 1])
            self._layers[k - 1].update_w(self._alpha, self._lambda, g_w)
            g_b = g_a
            if self._debug:
                b_shape = self._layers[k - 1]._b.shape
                g_b_debug = numpy.zeros((b_shape[1], b_shape[0]), dtype=numpy.float64)
                for i in range(b_shape[0]):
                    for j in range(b_shape[1]):
                        self._layers[k - 1]._b[i, j] += 0.001
                        self.forward_propagation(self._H[0])
                        Hl = -numpy.log(self._H[-1][y])
                        self._layers[k - 1]._b[i, j] -= 0.002
                        self.forward_propagation(self._H[0])
                        Hr = -numpy.log(self._H[-1][y])
                        g_b_debug[j, i] = (Hl - Hr) / 0.002
                        self._layers[k - 1]._b[i, j] += 0.001
                self.forward_propagation(self._H[0])
            self._layers[k - 1].update_b(self._alpha, self._lambda, g_b)
            #g_h = self._layers[k - 1].gradient_h(g_a)
            g_h = numpy.dot(g_a, self._layers[k - 1]._w.transpose())
            g_a = self._layers[k - 2].gradient_a(g_h, self._H[k - 2])

        # for the input layer
        g_w = self._layers[0].gradient_w(g_a, self._H[0])
        self._layers[0].update_w(self._alpha, self._lambda, g_w)
        g_b = g_a
        self._layers[0].update_b(self._alpha, self._lambda, g_b)

    def train(self, x_train, y_train, x_valid, y_valid, epoch):
        print('------------------ Start Training -----------------')
        print('\tepoch\t|\tscore\t')
        for j in range(epoch):
            print('\t', j, '\t', sep='', end=' ')
            shuffle_idx = numpy.arange(y_train.shape[0])
            numpy.random.shuffle(shuffle_idx)
            for i in range(y_train.size):
                idx = shuffle_idx[i]
                x = x_train[idx]
                x = x.reshape(x.shape[0], 1)
                if self._debug:
                    self.forward_propagation(x + 0.000001)
                    Hl = self._H[-1]
                    self.forward_propagation(x)
                    Hr = self._H[-1]
                self.forward_propagation(x)


                self.back_propagation(y_train[idx])
            # start a validation
            print('\t', self.score(x_valid, y_valid), '\t', sep='')

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

    def score(self, X, Y):
        Y_predicted = self.predict(X)
        correct_count = 0
        for i in range(Y.size):
            if Y[i] == Y_predicted[i]:
                correct_count += 1
        return (correct_count + 0.0) / float(Y.size)

class SingleLayerNetwork(NeuralNetwork):
    def __init__(self, input_dimension, hidden_dimension, output_dimension, debug=False):
        layer0 = SigmoidLayer(input_dimension, hidden_dimension)
        layer1 = SoftmaxLayer(hidden_dimension, output_dimension)
        layer_list = [layer0, layer1]
        NeuralNetwork.__init__(self, layer_list, debug=debug)




