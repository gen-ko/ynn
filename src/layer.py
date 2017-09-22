import numpy
import math
from src import ytensor


class Layer(object):
    def __init__(self, input_dimension, output_dimension):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension

    def forward(self, x):
        raise ValueError('Calling a virtual function')

    def backward(self, g_a, h_out, h_in):
        raise ValueError('Calling a virtual function')

    def update(self, learning_rate, regular, momentum):
        raise ValueError('Calling a virtual function')


class Linear(Layer):
    def __init__(self, input_dimension, output_dimension):
        Layer.__init__(self, input_dimension, output_dimension)
        b = math.sqrt(6.0) / math.sqrt(input_dimension + output_dimension + 0.0)
        self.w = numpy.random.uniform(low=-b, high=b, size=(input_dimension, output_dimension))
        self.b = numpy.zeros((output_dimension, 1), dtype=numpy.float64)
        self.delta_w = numpy.zeros(shape=self.w.shape, dtype=numpy.float64)
        self._delta_b = numpy.zeros(shape=self.b.shape, dtype=numpy.float64)
        self.g_w = numpy.zeros(self.w.shape)
        self.g_b = numpy.zeros(self.b.shape)

    def forward(self, x):
        tmp = numpy.dot(self.w.T, x)
        for i in range(tmp.shape[1]):
            tmp[:, i] += self.b[:, 0]
        return tmp

    def backward(self, g_a, h_out, h_in):
        self.g_w = numpy.zeros(self.w.shape)
        for i in range(g_a.shape[1]):
            self.g_w += numpy.outer(h_in[:, i], g_a[:, i])
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
        self.g_w = numpy.zeros(self.w.shape)
        self.g_b = numpy.zeros(self.b.shape)


class Nonlinear(Layer):
    def activation(self, a):
        raise ValueError('Calling a virtual function')

    def derivative(self, h):
        raise ValueError('Calling a virtual function')

    def backward(self, g_h, h_out, h_in):
        g_a = numpy.zeros(h_out.shape, dtype=numpy.float64)
        deri = self.derivative(h_out)
        for i in range(g_a.shape[0]):
            for j in range(g_a.shape[1]):
                g_a[i, j] = g_h[i, j] * deri[i, j]
        return g_a

    def forward(self, x):
        return self.activation(x)

    def update(self, learning_rate, regular, momentum):
        return

class Sigmoid(Nonlinear):
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


class FullConnectLayer(Layer):
    # assume current layer is the k'th layer

    def __init__(self, input_dimension, output_dimension):
        Layer.__init__(self, input_dimension, output_dimension)
        b = math.sqrt(6.0) / math.sqrt(input_dimension + output_dimension + 0.0)
        self.w = numpy.random.uniform(low=-b, high=b, size=(input_dimension, output_dimension))
        self.b = numpy.zeros((output_dimension, 1), dtype=numpy.float64)
        self.delta_w = numpy.zeros(shape=self.w.shape, dtype=numpy.float64)
        self._delta_b = numpy.zeros(shape=self.b.shape, dtype=numpy.float64)
        self.g_w = numpy.zeros(self.w.shape)
        self.g_b = numpy.zeros(self.b.shape)

    def activation(self, a):
        raise ValueError('Calling a virtual function')

    def derivative(self, h):
        raise ValueError('Calling a virtual function')

    def forward(self, x):
        tmp = numpy.dot(self.w.T, x)
        for i in range(tmp.shape[1]):
            tmp[:, i] += self.b[:, 0]
        return self.activation(tmp)

    def backward(self, g_h, h_out, h_in):
        g_a = numpy.zeros(h_out.shape, dtype=numpy.float64)
        deri = self.derivative(h_out)
        for i in range(g_a.shape[0]):
            for j in range(g_a.shape[1]):
                g_a[i, j] = g_h[i, j] * deri[i, j]
        self.g_w = numpy.zeros(self.w.shape)
        for i in range(g_a.shape[1]):
            self.g_w += numpy.outer(h_in[:, i], g_a[:, i])
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
        self.g_w = numpy.zeros(self.w.shape)
        self.g_b = numpy.zeros(self.b.shape)




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
            tmp[:, i] = numpy.exp(x[:, i])
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
            tmp = numpy.outer(h_in[:, i], g_a[:, i])
            self.g_w += tmp
            self.g_b += g_a[:, i].reshape(self.b.shape)
        self.g_w /= self.g_w.shape[1]
        self.g_b /= self.g_b.shape[1]
        g_h_in = numpy.dot(self.w, g_a)
        return g_h_in

class BatchNormalize(Layer):
    def __init__(self, in_dim, out_dim):
        Layer.__init__(self, in_dim, out_dim)

        b = math.sqrt(6.0) / math.sqrt(in_dim + out_dim + 0.0)
        self.gamma = numpy.random.uniform(low=-b, high=b)
        self.beta = 0.0
        self.delta_gamma = 0.0
        self._delta_beta = 0.0
        self.g_gamma = 0.0
        self.g_beta = 0.0
        self.xhat = 0.0
        self.xmu = 0.0
        self.ivar = 0.0
        self.sqrtvar = 0.0
        self.var = 0.0
        self.eps = 0.0
        self.dx = 0.0
        self.dgamma = 0.0
        self.dbeta = 0.0

    def forward(self, x):
        D, N = x.shape
        self.eps = 10.0**(-8)

        # step1: calculate mean
        mu = numpy.mean(x, axis=1)

        # step2: subtract mean vector of every trainings example
        self.xmu = (x.T - mu).T

        # step3: following the lower branch - calculation denominator
        sq = numpy.power(self.xmu, 2)

        # step4: calculate variance
        self.var = 1. / N * numpy.sum(sq, axis=1)

        # step5: add eps for numerical stability, then sqrt
        self.sqrtvar = numpy.sqrt(self.var + self.eps)

        # step6: invert sqrtwar
        self.ivar = 1. / self.sqrtvar

        # step7: execute normalization
        self.xhat = (self.xmu.T * self.ivar).T

        # step8: Nor the two transformation steps
        gammax = (self.gamma * self.xhat.T).T

        # step9
        out = (gammax.T + self.beta).T

        return out

    def backward(self, g_h, h_out, h_in):

        # get the dimensions of the input/output
        D, N = h_out.shape

        # step9
        self.dbeta = numpy.sum(h_out, axis=1)
        dgammax = h_out  # not necessary, but more understandable

        # step8
        self.dgamma = numpy.sum(dgammax * self.xhat, axis=1)
        dxhat = (dgammax.T * self.gamma).T

        # step7
        divar = numpy.sum(dxhat * self.xmu, axis=1)
        dxmu1 = (dxhat.T * self.ivar).T

        # step6
        dsqrtvar = -1. / (self.sqrtvar ** 2) * divar

        # step5
        dvar = 0.5 * 1. / numpy.sqrt(self.var + self.eps) * dsqrtvar

        # step4
        dsq = ((1. / N * numpy.ones((D, N))).T * dvar).T

        # step3
        dxmu2 = 2 * self.xmu * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * numpy.sum(dxmu1 + dxmu2, axis=1)

        # step1
        dx2 = (1. / N * numpy.ones((N, D)) * dmu).T

        # step0
        self.dx = dx1 + dx2
        return self.dx

    def update(self, learning_rate, regular, momentum):
        self.gamma += -learning_rate * self.dgamma
        self.beta += -learning_rate * self.dbeta