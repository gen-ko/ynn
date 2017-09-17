import numpy
import math


class Layer(object):
    @property
    def input_dimension(self):
        return self._input_dimension

    @property
    def output_dimension(self):
        return self._output_dimension

    def __init__(self, input_dimension, output_dimension):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension


class LinearLayer(Layer):
    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b

    def __init__(self, input_dimension, output_dimension):
        Layer.__init__(self, input_dimension, output_dimension)
        self._w = (numpy.random.rand(self.output_dimension, self.input_dimension) - 0.5) / 10.0
        self._b = (numpy.random.rand(self.output_dimension) - 0.5) / 10.0

    def activation(self, x):
        # x shape (x.dimension, )
        return numpy.matmul(self._w, x) + self._b


class InputLayer(LinearLayer):
    def __init__(self, input_dimension, output_dimension):
        LinearLayer.__init__(self, input_dimension, output_dimension)


class OutputLayer(LinearLayer):
    def __init__(self, input_dimension, output_dimension):
        LinearLayer.__init__(self, input_dimension, output_dimension)


class NonlinearLayer(Layer):
    def __init__(self, input_dimension, output_dimension):
        Layer.__init__(self, input_dimension, output_dimension)

    def single_dimension_activation(self, x):
        raise ValueError('Not calling an actual function')

    def activation(self, x):
        vfunc = numpy.vectorize(self.single_dimension_activation)
        return vfunc(x)


class SigmoidLayer(NonlinearLayer):
    def __init__(self, input_dimension, output_dimension):
        NonlinearLayer.__init__(self, input_dimension, output_dimension)

    def single_dimension_activation(self, x):
        return 1 / (1 + math.exp(-x))


class SoftmaxLayer(Layer):
    def __init__(self, input_dimension, output_dimension):
        Layer.__init__(self, input_dimension, output_dimension)

    def activation(self, x):
        expx = numpy.array([math.exp(xi) for xi in x])
        return expx / expx.sum()


class SingleLayerNetwork:
    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        self._inputLayer = InputLayer(input_dimension, hidden_dimension)
        self._hiddenLayer = SigmoidLayer(hidden_dimension, hidden_dimension)
        self._outputLayer = OutputLayer(hidden_dimension, output_dimension)
        self._softmaxLayer = SoftmaxLayer(output_dimension, output_dimension)

    def feed_forward(self, x):
        a = self._inputLayer.activation(x)
        h = self._hiddenLayer.activation(a)
        o = self._outputLayer.activation(h)
        s = self._softmaxLayer.activation(o)
        return s

    def back_propagation(self, y, learning_rate):
        return


