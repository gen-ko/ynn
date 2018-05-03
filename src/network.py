import src.layer as layer
import numpy as np
import src.util_functions as uf

class NeuralNetwork(object):
    def fprop(self, *args):
        raise ValueError('Calling a virtual function')

    def bprop(self, *args):
        raise ValueError('Calling a virtual function')

    def status_callback(self, *args):
        raise ValueError('Calling a virtual function')

    def plot_callback(self, *args):
        raise ValueError('Calling a virtual function')

    def update(self, train_settings):
        for layer_c in self.layers:
            layer_c.update(train_settings)
        return

    def dump(self):
        dump_obj = list()
        for layer_c in self.layers:
            dump_obj.append(layer_c.dump())
        return dump_obj

    def load(self, dump_obj: list):
        for layer_c in self.layers:
            layer_c.load(dump_obj.pop(0))
        return

# TODO: refactor the NeuralNetwork class


class MLP(NeuralNetwork):
    def __init__(self, layers):
        self.layers = layers
        self.h: list = None
        if __debug__:
            print('DEBUG MODE ENABLED')
        return

    def fprop(self, x, keep_state: bool=False):
        h = [x]
        hi = x
        for layer in self.layers:
            hi = layer.forward(hi)
            h.append(hi)
        if keep_state:
            self.h = h
        return h[-1]

    def bprop(self, y):
        d = [y]
        di = y
        for layer, hi in reversed(zip(self.layers, self.h)):
            di = layer.backward(di, self.h)

        d_h12 = self.layers[4].backward(y, self.h[13], self.h[12])
        d_h11 = self.layers[3].backward(d_h12, self.h[12], self.h[11])

        d_h10 = self.layers[2].backward(d_h11, self.h[11], self.h[10])
        d_h5, d_h9 = self.layers[1].backward(d_h10, self.h[10], self.h[5], self.h[9])

        #d_h8 = self.layers[2].backward(d_h9, self.h[9], self.h[8])
        #d_h4, d_h7 = self.layers[1].backward(d_h8, self.h[8], self.h[4], self.h[7])

        #d_h6 = self.layers[2].backward(d_h7, self.h[7], self.h[6])
        #d_h3, _ = self.layers[1].backward(d_h6, self.h[6], self.h[3], self.s0)

        #self.layers[0].backward(d_h3, self.h[3], self.h[0])
        #self.layers[0].backward(d_h4, self.h[4], self.h[1])
        self.layers[0].backward(d_h5, self.h[5], self.h[2])
        return