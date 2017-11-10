import numpy
import math
from src import ytensor
import warnings

warnings.filterwarnings('error')

name_pool: dict = {}


class Layer(object):
    def __init__(self, input_dimension, output_dimension, name: str='Base'):
        self.id = name_pool.__sizeof__()
        if name in name_pool:
            name_num = name_pool.__sizeof__()
            new_name = name + '_' + str(name_num)
            name_num += 1
            while new_name in name_pool:
                new_name = name + '_' + str(name_num)
            self.name = new_name
        else:
            self.name = name

        name_pool[self.name] = self
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

    def forward(self, x):
        raise ValueError('Calling a virtual function')

    def backward(self, *args):
        raise ValueError('Calling a virtual function')

    def update(self, *args):
        raise ValueError('Calling a virtual function')



class Linear(Layer):
    def __init__(self, input_dimension, output_dimension, name: str = 'Linear'):
        Layer.__init__(self, input_dimension, output_dimension, name)
        b = math.sqrt(6.0) / math.sqrt(input_dimension + output_dimension + 0.0)
        self.w = numpy.random.uniform(low=-b, high=b, size=(input_dimension, output_dimension)).astype(numpy.float32)
        self.b = numpy.zeros((output_dimension, ), dtype=numpy.float32)
        self.delta_w = numpy.zeros(shape=self.w.shape, dtype=numpy.float32)
        self.delta_b = numpy.zeros(shape=self.b.shape, dtype=numpy.float32)
        self.d_w = numpy.zeros(self.w.shape, dtype=numpy.float32)
        self.d_b = numpy.zeros(self.b.shape, dtype=numpy.float32)

    def forward(self, x):
        tmp = numpy.dot(x, self.w)
        tmp += self.b
        return tmp

    def backward(self, d_a, h_out, h_in):
        # slow, 5s
        batch_size = d_a.shape[0]
        self.d_w = numpy.tensordot(h_in, d_a, axes=(0, 0))
        self.d_b = numpy.sum(d_a, axis=0)
        #for i in range(batch_size):
        #    self.d_w += numpy.outer(h_in[i], d_a[i])
        #    self.d_b += d_a[i]
        self.d_w /= batch_size
        self.d_b /= batch_size
        d_h_in = numpy.dot(d_a, self.w.T)
        return d_h_in

    def update(self, learning_rate, regular=0.0, momentum=0.0):
        tmp = self.d_w + regular * 2.0 * self.w
        self.delta_w = -learning_rate * tmp + momentum * self.delta_w
        self.w += self.delta_w

        tmp = self.d_b
        self.delta_b = -learning_rate * tmp + momentum * self.delta_b
        self.b += self.delta_b
        self.d_w = numpy.zeros(self.w.shape)
        self.d_b = numpy.zeros(self.b.shape)

        #self.w -= self.d_w * learning_rate
        #self.b -= self.d_b * learning_rate
        return


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

        #tmp = numpy.clip(tmp, -500.0, 500.0)
        #tmp = numpy.exp(-tmp) + 1
        #tmp = numpy.reciprocal(tmp)
        tmp = ytensor.sigmoid(tmp)
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

        #tmp = numpy.clip(tmp, -500.0, 500.0)
        #tmp = numpy.exp(-tmp) + 1
        #tmp = numpy.reciprocal(tmp)
        tmp = ytensor.sigmoid(tmp)
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
    def __init__(self, dimension, name: str = 'Nonlinear'):
        Layer.__init__(self, dimension, dimension, name)
        return

    def activation(self, a):
        raise ValueError('Calling a virtual function')

    def derivative(self, h):
        raise ValueError('Calling a virtual function')

    def backward(self, d_h_out, h_out, h_in):
        deri = self.derivative(h_out)
        d_h_in = numpy.multiply(d_h_out, deri)
        return d_h_in

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
        tmp = numpy.multiply(tmp, h_out)
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


# modified to conform to the new input layout, (batch_size, dimension)
class Softmax(Layer):
    def __init__(self, dimension, name: str = 'Softmax'):
        Layer.__init__(self, dimension, dimension, name)
        return

    def forward(self, x):
        tmp = numpy.clip(x, -100, 100)
        tmp = numpy.exp(tmp)
        tmp2 = numpy.sum(tmp, axis=1)
        tmp = tmp.T
        tmp /= tmp2
        return tmp.T

    def backward(self, y, h_out, h_in):
        batch_size = h_out.shape[0]
        for i in range(batch_size):
            h_out[i, y[i]] -= 1.0
        return h_out

    def update(self, *args):
        return



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


class Embedding(Layer):
    def __init__(self, input_dimension, output_dimension, name: str = 'Embedding'):
        Layer.__init__(self, input_dimension, output_dimension, name)
       # b = math.sqrt(6.0) / math.sqrt(input_dimension + output_dimension + 0.0)
        self.d_w = None
        self.d_w_index = None
       #self.w = numpy.random.uniform(low=-b, high=b, size=(input_dimension, output_dimension))
       #self.w = numpy.array(self.w, dtype=numpy.float32)
        self.w = numpy.random.normal(0.0, 1.0, size=(input_dimension, output_dimension)).astype(numpy.float32)

    def forward(self, x):
        return self.w[x]

    def backward(self, d_h, h_out, h_in):
        batch_size = h_out.shape[0]
        try:
            self.d_w = numpy.append(self.d_w, d_h / batch_size, axis=0)
            self.d_w_index = numpy.append(self.d_w_index, h_in, axis=0)
        except:
            self.d_w = d_h / batch_size
            self.d_w_index = h_in
        return

    def update(self, learning_rate, *args):
        # batch_size = self.d_w_index.size
        self.w[self.d_w_index] -= learning_rate * self.d_w
        self.d_w_index = None
        self.d_w = None
        return


