import numpy
import math
from src import ytensor

class Layer(object):
    def __init__(self, input_dimension, output_dimension):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension

    def forward(self, x):
        raise ValueError('Calling a virtual function')


class FullConnectLayer(Layer):
    # assume current layer is the k'th layer

    def __init__(self, input_dimension, output_dimension):
        Layer.__init__(self, input_dimension, output_dimension)
        b = math.sqrt(6.0) / math.sqrt(input_dimension + output_dimension + 0.0)
        self.w = numpy.random.uniform(low=-b, high=b, size=(input_dimension, output_dimension))
        self.b = numpy.zeros((output_dimension, 1), dtype=numpy.float64)
        self.delta_w = numpy.zeros(shape=self.w.shape, dtype=numpy.float64)
        self._delta_b = numpy.zeros(shape=self.b.shape, dtype=numpy.float64)
        self.g_w: numpy.ndarray
        self.g_b: numpy.ndarray

    def activation(self, a):
        raise ValueError('Calling a virtual function')

    def derivative(self, h):
        raise ValueError('Calling a virtual function')

    def forward(self, x):
        tmp = numpy.dot(self.w.T, x)
        for i in range(tmp.shape[1]):
            tmp[:, i] += self.b
        return self.activation(tmp)

    def backward(self, g_h, h_out, h_in):
        g_a = numpy.zeros(h_out.shape, dtype=numpy.float64)
        deri = self.derivative(h_out)
        for i in range(g_a.shape[0]):
            for j in range(g_a.shape[1]):
                g_a[i, j] = g_h[i, h_out] * deri[i, j]
        self.g_w = numpy.zeros(self.w.shape)
        for i in range(g_a.shape[1]):
            self.g_w += numpy.outer(g_a[:, i], h_in[:, i])
            self.g_b += g_a[:, i].reshape(self.b.shape)
        self.g_w /= self.g_w.shape[1]
        self.g_b /= self.g_b.shape[1]
        g_h_in = numpy.dot(self.w, g_a)
        return g_h_in

    def update(self, learning_rate, regular, momentum):
        self.delta_w = self.g_w + momentum * self.delta_w
        delta = -self.delta_w - regular * 2.0 * self.w
        self.w += learning_rate * delta
        self._delta_b = self.g_b + momentum * self._delta_b
        delta = -self._delta_b
        self.b += learning_rate * delta




class SigmoidLayer(FullConnectLayer):
    def __init__(self, input_dimension, output_dimension):
        FullConnectLayer.__init__(self, input_dimension, output_dimension)

    def activation(self, x):
        x = numpy.clip(x, -500.0, 500.0)
        tmp = numpy.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                tmp[i, j] = 1.0 / (1.0 + math.exp(-x[i, j]))
        return tmp

    def derivative(self, h):
        tmp = numpy.zeros(h.shape)
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                tmp[i, j] = h[i, j] * (1 - h[i, j])
        return tmp

# TODO
# TODO class ReLULayer(FullConnectLayer):
        # TODO    def __init__(self, input_dimension, output_dimension):
    # TODO        FullConnectLayer.__init__(self, input_dimension, output_dimension)
    # TODO
        # TODO   def activation(self, x):
    # TODO        return numpy.array([max(xi, 0) for xi in x]).reshape(x.shape)

        # TODO   def derivative(self, h):
    # TODO      return numpy.array([max(numpy.sign(hi), 0) for hi in h])


    # TODO class TanhLayer(FullConnectLayer):
        # TODO  def __init__(self, input_dimension, output_dimension):
    # TODO       FullConnectLayer.__init__(self, input_dimension, output_dimension)
    # TODO
        # TODO    def activation(self, x):
    # TODO       return numpy.tanh(x).reshape(x.shape)

        # TODO   def derivative(self, h):
    # TODO      return numpy.array([1.0 - hi**2 for hi in h])


class SoftmaxLayer(FullConnectLayer):
    def __init__(self, input_dimension, output_dimension):
        FullConnectLayer.__init__(self, input_dimension, output_dimension)

    def activation(self, x):
        tmp = numpy.zeros(x.shape)
        for i in range(tmp.shape[1]):
            tmp[:, i] = numpy.array([math.exp(xi) for xi in x])
            tmp[:, i] = tmp[:, i] / numpy.sum(tmp[:, i])
        return tmp

    def derivative(self, h):
        raise ValueError('''Softmax Layer doesn't need derivative''')

    def backward(self, y, h_out, h_in):
        g_a = numpy.array(h_out)
        for i in range(g_a.shape[1]):
            g_a[y[i], i] -= 1.0

        self.g_w = numpy.zeros(self.w.shape)
        for i in range(g_a.shape[1]):
            self.g_w += numpy.outer(g_a[:, i], h_in[:, i])
            self.g_b += g_a[:, i].reshape(self.b.shape)
        self.g_w /= self.g_w.shape[1]
        self.g_b /= self.g_b.shape[1]
        g_h_in = numpy.dot(self.w, g_a)
        return g_h_in



