import numpy
import math

class Layer(object):
    def __init__(self, input_dimension, output_dimension):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension

    def forward_propagation(self):
        raise ValueError('Calling a virtual function')

    def gradients(self):
        raise ValueError('Calling a virtual function')

    def update(self):
        raise ValueError('Calling a virtual function')

class FullConnectLayer(Layer):
    # assume current layer is the k'th layer

    def __init__(self, input_dimension, output_dimension):
        Layer.__init__(self, input_dimension, output_dimension)
        # w shall be draw from [-b, b]
        b = math.sqrt(6.0) / math.sqrt(input_dimension + output_dimension + 0.0)
        self.w = numpy.random.uniform(low=-b, high=b, size=(input_dimension, output_dimension))
        # the shape of formal weights w, where a = w^T x + b
        # self._w is of shape (output_dimension, input_dimension)
        self._w = numpy.random.uniform(low=-b, high=b, size=(output_dimension, input_dimension))
        self.b = numpy.zeros((output_dimension, 1), dtype=numpy.float64)
        # formal b is of shape (output_dimension, )
        self._b = numpy.zeros((output_dimension, ), dtype=numpy.float64)
        # momentum can be modified per run of training
        self.delta_w = numpy.zeros(shape=self.w.shape, dtype=numpy.float64)
        self._delta_b = numpy.zeros(shape=self.b.shape, dtype=numpy.float64)

    def activation(self, a):
        raise ValueError('Calling a virtual function')

    def derivative(self, h):
        raise ValueError('Calling a virtual function')

    def forward_slow(self, x):
        return self.activation(numpy.dot(self.w.T, x) + self.b)
                   
    def forward_single(self, x):
        return self.activation(numpy.dot(self._w, x) + self._b)
    
    def forward_batch(self, x):
        return self.activation(numpy.dot(self._w, x) + numpy.tile(self._b, (x.shape[1], 1).T))

    def gradient_a(self, gradient_h, h):
        g_a = numpy.zeros((1, self._output_dimension), dtype=numpy.float64)
        deri = self.derivative(h)
        for i in range(deri.shape[0]):
            g_a[0, i] = deri[i] * gradient_h[0, i]
        return g_a
                   
    def gradient(self):
        return

    def update_w(self, g_w, learning_rate, regular, momentum):
        self.delta_w = g_w.transpose() + momentum * self.delta_w
        delta = -self.delta_w - regular * 2.0 * self.w
        self.w += learning_rate * delta

    def update_b(self, g_b, learning_rate, regular, momentum):
        self._delta_b = g_b.transpose() + momentum * self._delta_b
        delta = -self._delta_b
        self.b += learning_rate * delta
        
    def update(self, delta_w, delta_b):
        self._w += delta_w
        self._b += delta_b


class SigmoidLayer(FullConnectLayer):
    def __init__(self, input_dimension, output_dimension):
        FullConnectLayer.__init__(self, input_dimension, output_dimension)

    def activation(self, x):
        x = numpy.clip(x, -500.0, 500.0)
        return numpy.array([1.0 / (1.0 + math.exp(-xi)) for xi in x]).reshape(x.shape)

    def derivative(self, h):
        return numpy.array([hi * (1 - hi) for hi in h])


class ReLULayer(FullConnectLayer):
    def __init__(self, input_dimension, output_dimension):
        FullConnectLayer.__init__(self, input_dimension, output_dimension)

    def activation(self, x):
        return numpy.array([max(xi, 0) for xi in x]).reshape(x.shape)

    def derivative(self, h):
        return numpy.array([max(numpy.sign(hi), 0) for hi in h])


class TanhLayer(FullConnectLayer):
    def __init__(self, input_dimension, output_dimension):
        FullConnectLayer.__init__(self, input_dimension, output_dimension)

    def activation(self, x):
        return numpy.tanh(x).reshape(x.shape)

    def derivative(self, h):
        return numpy.array([1.0 - hi**2 for hi in h])


class SoftmaxLayer(FullConnectLayer):
    def __init__(self, input_dimension, output_dimension):
        FullConnectLayer.__init__(self, input_dimension, output_dimension)

    def activation(self, x):
        expi = numpy.array([math.exp(xi) for xi in x])
        return (expi / numpy.sum(expi)).reshape(x.shape[0], 1)

    def derivative(self, h):
        raise ValueError('''Softmax Layer doesn't need derivative''')



