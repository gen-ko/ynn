# Created by Yuan Liu on Sep/21/17
import numpy


def dot_23(x: numpy.ndarray, y: numpy.ndarray):
    return numpy.array([numpy.dot(x, y[i, :, :]) for i in range(x.shape[2])])


def dot_32(x: numpy.ndarray, y: numpy.ndarray):
    return numpy.array([numpy.dot(x[i, :, :], y) for i in range(x.shape[2])])