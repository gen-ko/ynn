import numpy
import math
from src import ytensor
import warnings

warnings.filterwarnings('error')


class Layer(object):
    def __init__(self, input_dimension, output_dimension):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension

    def forward(self, x):
        raise ValueError('Calling a virtual function')

    def backward(self, *args):
        raise ValueError('Calling a virtual function')

    def update(self, *args):
        raise ValueError('Calling a virtual function')


class Linear(Layer):
    def __init__(self, input_dimension, output_dimension):
        Layer.__init__(self, input_dimension, output_dimension)
        b = math.sqrt(6.0) / math.sqrt(input_dimension + output_dimension + 0.0)
        self.w = numpy.random.uniform(low=-b, high=b, size=(input_dimension, output_dimension))
        self.b = numpy.zeros((output_dimension, 1), dtype=numpy.float64)
        self.delta_w = numpy.zeros(shape=self.w.shape, dtype=numpy.float64)
        self.delta_b = numpy.zeros(shape=self.b.shape, dtype=numpy.float64)
        self.g_w = numpy.zeros(self.w.shape)
        self.g_b = numpy.zeros(self.b.shape)

    def forward(self, x):
        tmp = numpy.dot(self.w.T, x)
        # tmp shape (D, N), b shape (D, 1)
        tmp += self.b
        return tmp

    def backward(self, g_a, h_out, h_in):
        for i in range(g_a.shape[1]):
            self.g_w += numpy.outer(h_in[:, i], g_a[:, i])
            self.g_b += g_a[:, i].reshape(self.b.shape)
        self.g_w /= g_a.shape[1]
        self.g_b /= g_a.shape[1]
        g_h_in = numpy.dot(self.w, g_a)
        return g_h_in

    def update(self, learning_rate, regular, momentum):
        tmp = self.g_w + regular * 2.0 * self.w
        self.delta_w = -learning_rate * tmp + momentum * self.delta_w
        self.w += self.delta_w

        tmp = self.g_b
        self.delta_b = -learning_rate * tmp + momentum * self.delta_b
        self.b += self.delta_b
        self.g_w = numpy.zeros(self.w.shape)
        self.g_b = numpy.zeros(self.b.shape)


class RBM(Layer):
    def __init__(self, input_dimension, output_dimension):
        Layer.__init__(self, input_dimension, output_dimension)
        b = math.sqrt(6.0) / math.sqrt(input_dimension + output_dimension + 0.0)
        self.w = numpy.random.uniform(low=-b, high=b, size=(output_dimension, input_dimension))
        self.h_bias = numpy.zeros(shape=(output_dimension, 1), dtype=numpy.float64)
        self.x_bias = numpy.zeros(shape=(input_dimension, 1), dtype=numpy.float64)

    def forward(self, x):
        tmp = numpy.dot(self.w, x)
        tmp += self.h_bias

        tmp = numpy.clip(tmp, -500.0, 500.0)
        tmp = numpy.exp(-tmp) + 1
        tmp = numpy.reciprocal(tmp)
        return tmp

    def sample_h_given_x(self, x):
        h_mean = self.forward(x)
        h_sample = numpy.random.binomial(n=1, p=h_mean, size=h_mean.shape)
        return h_sample

    def sample_x_given_h(self, h):
        x_mean = self.backward(h)
        x_sample = numpy.random.binomial(n=1, p=x_mean, size=x_mean.shape)
        return x_sample

    def gibbs_xhx(self, x):
        h_sample = self.sample_h_given_x(x)
        x_sample = self.sample_x_given_h(h_sample)
        return x_sample

    def gibbs_hxh(self, h):
        x_sample = self.sample_x_given_h(h)
        h_sample = self.sample_h_given_x(x_sample)
        return h_sample

    def backward(self, h):
        tmp = numpy.dot(self.w.T, h)
        tmp += self.x_bias

        tmp = numpy.clip(tmp, -500.0, 500.0)
        tmp = numpy.exp(-tmp) + 1
        tmp = numpy.reciprocal(tmp)
        return tmp

    def update(self, delta_w, delta_h_bias, delta_x_bias, learning_rate):
        self.w += learning_rate * delta_w
        self.h_bias += learning_rate * delta_h_bias
        self.x_bias += learning_rate * delta_x_bias


class AutoEncoder(Layer):
    def __init__(self, input_dimension, output_dimension):
        Layer.__init__(self, input_dimension, output_dimension)
        #b = math.sqrt(6.0) / math.sqrt(input_dimension + output_dimension + 0.0)
        #self.w = numpy.random.uniform(low=-b, high=b, size=(output_dimension, input_dimension))
        self.w = numpy.random.normal(0, 0.1, (output_dimension, input_dimension))
        self.h_bias = numpy.zeros(shape=(output_dimension, 1), dtype=numpy.float64)
        self.x_bias = numpy.zeros(shape=(input_dimension, 1), dtype=numpy.float64)

    def forward(self, x):
        tmp = numpy.dot(self.w, x)
        tmp += self.h_bias

        tmp = numpy.clip(tmp, -500.0, 500.0)
        tmp = numpy.exp(-tmp) + 1
        tmp = numpy.reciprocal(tmp)
        return tmp

    def backward(self, h):
        tmp = numpy.dot(self.w.T, h)
        tmp += self.x_bias

        tmp = numpy.clip(tmp, -500.0, 500.0)
        tmp = numpy.exp(-tmp) + 1
        tmp = numpy.reciprocal(tmp)
        return tmp

    def update(self, delta_w, delta_h_bias, delta_x_bias, learning_rate):
        self.w += learning_rate * delta_w
        self.h_bias += learning_rate * delta_h_bias
        self.x_bias += learning_rate * delta_x_bias


class Dropout(Layer):
    def __init__(self, input_dimension, output_dimension, drop_rate=0.1):
        Layer.__init__(self, input_dimension, output_dimension)
        assert input_dimension == output_dimension, 'input and output dimension is not equal'
        self.drop_rate = drop_rate

    def forward(self, x):
        y = numpy.array(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if numpy.random.uniform(0, 1, ) < self.drop_rate:
                    y[i, j] = 0
        return y

    def backward(self, *args):
        return args




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
    def activation(self, x: numpy.ndarray):
        tmp = numpy.clip(x, -500.0, 500.0)
        tmp = numpy.exp(-tmp) + 1
        tmp = numpy.reciprocal(tmp)
        return tmp

    def derivative(self, h_out: numpy.ndarray):
        tmp = 1.0 - h_out
        tmp = numpy.multiply(tmp , h_out)
        return tmp


class ReLU(Nonlinear):
    def activation(self, x: numpy.ndarray):
        tmp = numpy.maximum(0.0, x)
        return tmp

    def derivative(self, h_out: numpy.ndarray):
        tmp = h_out > 0
        tmp = tmp.astype(numpy.float64)
        return tmp


class Tanh(Nonlinear):
    def activation(self, x: numpy.ndarray):
        tmp = numpy.tanh(x)
        return tmp

    def derivative(self, h_out: numpy.ndarray):
        tmp = numpy.power(h_out, 2)
        tmp = 1 - tmp
        return tmp


class Softmax(Layer):
    def forward(self, x):
        tmp = numpy.zeros(x.shape)
        for i in range(tmp.shape[1]):
            tmp[:, i] = numpy.exp(x[:, i])
            tmp[:, i] = tmp[:, i] / numpy.sum(tmp[:, i])
        return tmp

    def backward(self, y, h_out, h_in):
        g_a = numpy.array(h_out)
        for i in range(g_a.shape[1]):
            g_a[y[i], i] -= 1.0
        return g_a

    def update(self, learning_rate, regular, momentum):
        return


class FullConnectLayer(Layer):
    # assume current layer is the k'th layer

    def __init__(self, input_dimension, output_dimension):
        Layer.__init__(self, input_dimension, output_dimension)
        b = math.sqrt(6.0) / math.sqrt(input_dimension + output_dimension + 0.0)
        self.w = numpy.random.uniform(low=-b, high=b, size=(input_dimension, output_dimension))
        self.b = numpy.zeros((output_dimension, 1), dtype=numpy.float64)
        self.delta_w = numpy.zeros(shape=self.w.shape, dtype=numpy.float64)
        self.delta_b = numpy.zeros(shape=self.b.shape, dtype=numpy.float64)
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
        tmp = self.g_w + regular * 2.0 * self.w
        self.delta_w = -learning_rate * tmp + momentum * self.delta_w
        self.w += self.delta_w

        tmp = self.g_b
        self.delta_b = -learning_rate * tmp + momentum * self.delta_b
        self.b += self.delta_b
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
        for i in range(g_a.shape[1]):
            tmp = numpy.outer(h_in[:, i], g_a[:, i])
            self.g_w += tmp
            self.g_b += g_a[:, i].reshape(self.b.shape)
        self.g_w /= y.size
        self.g_b /= y.size
        g_h_in = numpy.dot(self.w, g_a)
        return g_h_in

class BN(Layer):
    def __init__(self, in_dim, out_dim):
        Layer.__init__(self, in_dim, out_dim)
        self.eps = 1e-5
        b = math.sqrt(6.0) / math.sqrt(in_dim + out_dim + 0.0)
        self.gamma = 1.0
        self.beta = 0.0
        self.delta_gamma = 0.0
        self.delta_b = 0.0
        self.g_g = 0.0
        self.g_b = 0.0

        self.x_hat : numpy.ndarray
        self.ivar = 0.0


    def forward(self, x):
        D, N = x.shape
        tmp_x = x.T
        mu = numpy.mean(tmp_x, axis=0)
        xmu = tmp_x - mu
        sq = xmu ** 2
        var = 1. / N * numpy.sum(sq, axis=0)
        sqrtvar = numpy.sqrt(var + self.eps)
        self.ivar = 1. / sqrtvar
        self.x_hat = xmu * self.ivar
        tmp = self.gamma * self.x_hat
        out = tmp + self.beta
        return out.T

    def backward(self, g_h, h_out, h_in):

        # get the dimensions of the input/output
        D, N = g_h.shape
        dout = g_h.T
        x_hat = self.x_hat
        inv_var = self.ivar


        dxhat = dout * self.gamma
        tmp1 = (1. / N) * self.ivar
        tmp2 = (N * dxhat - numpy.sum(dxhat, axis=0))
        tmp3 = (x_hat * numpy.sum(dxhat * x_hat, axis=0))
        dx = tmp1 * (tmp2 - tmp3)
        self.g_b = numpy.sum(dout, axis=0)
        self.g_g = numpy.sum(numpy.multiply(x_hat, dout), axis=0)

        return dx.T



    def update(self, learning_rate, regular, momentum):
        tmp = self.g_g + regular * 2.0 * self.gamma
        self.delta_gamma = -learning_rate * tmp + momentum * self.delta_gamma
        self.gamma += self.delta_gamma

        tmp = self.g_b
        self.delta_b = -learning_rate * tmp + momentum * self.delta_b
        self.beta += self.delta_b
        self.g_g = 0.0
        self.g_b = 0.0
