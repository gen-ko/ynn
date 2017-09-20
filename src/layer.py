import numpy
import math



class Layer(object):
    # assume current layer is the k'th layer

    def __init__(self, input_dimension, output_dimension, momentum=0.9):
        self.input_dimension = input_dimension
        # a read-only version
        self._input_dimension = input_dimension
        self.output_dimension = output_dimension
        # a read-only version
        self._output_dimension = output_dimension
        # w shall be draw from [-b, b]
        b = math.sqrt(6.0) / math.sqrt(input_dimension + output_dimension + 0.0)
        self.w = (numpy.random.rand(input_dimension, output_dimension) - 0.5) * 2 * b
        # the shape of formal weights w, where a = w^T x + b
        self._w_shape = (numpy.random.rand(output_dimension, input_dimension) - 0.5) * 2 * b
        # self._w is of shape (output_dimension, input_dimension)
        self._w = (numpy.random.rand(output_dimension, input_dimension) - 0.5) * 2
        self.b = numpy.zeros((output_dimension, 1), dtype=numpy.float64)
        # formal b is of shape (output_dimension, )
        self._b = numpy.zeros((output_dimension, ), dtype=numpy.float64)
        # momentum can be modified per run of trainning
        self.momentum = momentum
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
        for i in range(deri.shape[0]):
            g_a[0, i] = deri[i] * gradient_h[0, i]
        return g_a

    def update_w(self, learning_rate, regular, g_w):
        self.delta_w = g_w.transpose() + self.momentum * self.delta_w
        delta = -self.delta_w - regular * 2.0 * self.w
        self.w += learning_rate * delta

    def update_b(self, learning_rate, regular, g_b):
        self._delta_b = g_b.transpose() + self.momentum * self._delta_b
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



