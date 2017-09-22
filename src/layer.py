import numpy
import math
from src import ytensor


class Layer(object):
    def __init__(self, input_dimension, output_dimension):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension

    def forward(self, x):
        raise ValueError('Calling a virtual function')

    def backward(self, h_in, h_out, gradient_h):
        raise ValueError('Calling a virtual function')

    def gradients(self):
        raise ValueError('Calling a virtual function')

    def update(self):
        raise ValueError('Calling a virtual function')


class FullConnected(Layer):
    def initialize(self):
        b = math.sqrt(6.0) / math.sqrt(self._input_dimension + self._output_dimension + 0.0)
        self._w = (numpy.random.uniform(low=-b, high=b, size=(self._input_dimension, self._output_dimension)))
        self._b = (numpy.zeros((self._output_dimension, 1), dtype=numpy.float64))
        self._delta_w = (numpy.zeros(shape=self._w.shape, dtype=numpy.float64))
        self._delta_b = (numpy.zeros(shape=self._b.shape, dtype=numpy.float64))

    def set_hyperparameter(self, learning_rate=0.01, momentum=0.9, regularizer=0.0001):
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._regularizer = regularizer * 2

    def activation(self, a):
        raise ValueError('Calling a virtual function')

    def derivative(self, h):
        raise ValueError('Calling a virtual function')

    def forward(self, x):
        return self.activation(ytensor.fma_232(self._w.T, x, self._b))

    def backward(self, h_out, h_in, gradient_h):
        gradient_a = self.gradient_a(h_out, gradient_h)
        self._gradient_w = ytensor.dot_33(gradient_a.swapaxes(1, 2), h_in.swapaxes(1, 2))
        self._gradient_b = gradient_a.swapaxes(1, 2)
        return ytensor.dot_32(gradient_a, self._w.T)

    def gradient_a(self, h, gradient_h):
        return ytensor.inner_33(gradient_h, self.derivative(h))

    def update(self):
        avg_gradient_w = numpy.average(self._gradient_w, axis=0).T
        self._delta_w = -(avg_gradient_w + self._w * self._regularizer) * self._learning_rate + self._delta_w * self._momentum
        self._w += self._delta_w
        self._delta_b = -(numpy.average(self._gradient_b, axis=0)) * self._learning_rate +\
                        self._delta_b * self._momentum
        self._b += self._delta_b


class Sigmoid(FullConnected):
    def activation(self, x):
        x = numpy.clip(x, -500.0, 500.0)
        return numpy.vectorize(Sigmoid.activation_scalar)(x)

    def derivative(self, h):
        return numpy.array([numpy.array([hii * (1 - hii) for hii in hi]) for hi in h]).reshape(h.shape)

    @staticmethod
    def activation_scalar(x):
        return 1.0 / (1.0 + math.exp(-x))


class Softmax(Layer):
    def initialize(self):
        b = math.sqrt(6.0) / math.sqrt(self._input_dimension + self._output_dimension + 0.0)
        self._w =numpy.random.uniform(low=-b, high=b, size=(self._input_dimension, self._output_dimension))
        self._b = numpy.zeros((self._output_dimension, 1), dtype=numpy.float64)
        self._delta_w = (numpy.zeros(shape=self._w.shape, dtype=numpy.float64))
        self._delta_b = (numpy.zeros(shape=self._b.shape, dtype=numpy.float64))

    def set_hyperparameter(self, learning_rate=0.01, momentum=0.9, regularizer=0.0001):
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._regularizer = regularizer * 2

    @staticmethod
    def activation(a):
        expi = (numpy.exp(a))
        return expi / numpy.sum(expi)

    def derivative(self, h):
        raise ValueError('Calling a virtual function')

    def forward(self, x):
        return self.activation(ytensor.fma_232(self._w.T, x, self._b))

    def backward(self, h_out, h_in, y):
        gat = numpy.array(h_out)
        for i in range(y.shape[0]):
            gat[i, y[i], :] -= 1.0
        gradient_a = gat.swapaxes(1, 2)
        self._gradient_w = ytensor.dot_33(gradient_a.swapaxes(1, 2), h_in.swapaxes(1, 2))
        self._gradient_b = gradient_a.swapaxes(1, 2)
        return ytensor.dot_32(gradient_a, self._w.T)

    def update(self):
        avg_gradient_w = numpy.average(self._gradient_w, axis=0).T
        self._delta_w = -(avg_gradient_w + self._w * self._regularizer) * self._learning_rate + self._delta_w * self._momentum
        self._w += self._delta_w
        self._delta_b = -(numpy.average(self._gradient_b, axis=0)) * self._learning_rate +\
                        self._delta_b * self._momentum
        self._b += self._delta_b



