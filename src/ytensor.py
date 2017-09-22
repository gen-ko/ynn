# Created by Yuan Liu on Sep/21/17
import numpy


class Yarray(numpy.array):
    @property
    def T(self):
        return Yarray([self[i, :, :].T for i in range(self.shape[0])])


def dot_23(x: Yarray, y: Yarray):
    return Yarray([numpy.dot(x, y[i, :, :]) for i in range(y.shape[0])])


def dot_32(x: Yarray, y: Yarray):
    return Yarray([numpy.dot(x[i, :, :], y) for i in range(x.shape[0])])


def dot_33(x: Yarray, y: Yarray):
    return Yarray([numpy.dot(x[i, :, :], y[i, :, :]) for i in range(x.shape[0])])


def transpose(x: Yarray):
    return Yarray([x[i, :, :].T for i in range(x.shape[0])])


def fma_232(a: Yarray, b: Yarray, c: Yarray):
    return Yarray([numpy.dot(a, b[i, :, :]) + c for i in range(b.shape[0])])


def inner_23(x: Yarray, y: Yarray):
    return Yarray([numpy.inner(x, y[i, :, :]) for i in range(y.shape[0])])


def inner_32(x: Yarray, y: Yarray):
    return Yarray([numpy.inner(x[i, :, :], y) for i in range(x.shape[0])])


def inner_33(x: Yarray, y: Yarray):
    return Yarray([numpy.inner(x[i, :, :], y[i, :, :]) for i in range(x.shape[0])])



