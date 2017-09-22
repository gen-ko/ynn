# Created by Yuan Liu on Sep/21/17
import numpy


def transpose(x: numpy.ndarray):
    return numpy.array([x[i, :, :].T for i in range(x.shape[0])])

def dot_23(x: numpy.ndarray, y: numpy.ndarray):
    return numpy.array([numpy.dot(x, y[i, :, :]) for i in range(y.shape[0])])


def dot_32(x: numpy.ndarray, y: numpy.ndarray):
    return numpy.array([numpy.dot(x[i, :, :], y) for i in range(x.shape[0])])


def dot_33(x: numpy.ndarray, y: numpy.ndarray):
    return numpy.array([numpy.dot(x[i, :, :], y[i, :, :]) for i in range(x.shape[0])])


def transpose(x: numpy.ndarray):
    return numpy.array([x[i, :, :].T for i in range(x.shape[0])])


def fma_232(a: numpy.ndarray, b: numpy.ndarray, c: numpy.ndarray):
    return numpy.array([numpy.dot(a, b[i, :, :]) + c for i in range(b.shape[0])])

def inner_23(x: numpy.ndarray, y: numpy.ndarray):
    return numpy.array([numpy.inner(x, y[i, :, :]) for i in range(y.shape[0])])


def inner_32(x: numpy.ndarray, y: numpy.ndarray):
    return numpy.array([numpy.inner(x[i, :, :], y) for i in range(x.shape[0])])


def inner_33(x: numpy.ndarray, y: numpy.ndarray):
    z = numpy.array(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            z[i, 0, j] = x[i, 0, j] * y[i, j, 0]
    return z


def clip(a, a_min, a_max, out=None):
    return numpy.array(numpy.clip(a, a_min, a_max, out))


