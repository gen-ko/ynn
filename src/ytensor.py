# Created by Yuan Liu on Sep/21/17
import numpy
from scipy.special import expit

def fma(x: numpy.ndarray, w: numpy.ndarray, b: numpy.ndarray):
    # x : (sample_num, d1), w : (d2, d1), b : (d2, )
    return numpy.dot(w, x) + b

def sigmoid(x: numpy.ndarray):
    return expit(x)

def gibbs_sampling(x: numpy.ndarray, k, w):
    # x shape (sample_dimension, 1)
    h_t_prob = sigmoid(fma(x, w, b))
    h_t_sample = self.sample_h(h_t_prob)
    h_neg_prob = numpy.array(h_t_prob)
    h_neg_sample = numpy.array(h_t_sample)
    for ki in range(0, 1, 1):
        x_neg_prob = self.layer.backward(h_neg_sample)
        x_neg_sample = self.sample_x(x_neg_prob)
        h_neg_prob = self.layer.forward(x_neg_sample)
        h_neg_sample = self.sample_h(h_neg_prob)
    return [x_neg_prob, x_neg_sample, h_t_prob, h_neg_prob, h_neg_sample]

