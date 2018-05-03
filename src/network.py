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
        # h: 0, 1, ..., L, L+1
        for layer in self.layers:
            hi = layer.forward(hi)
            h.append(hi)
        if keep_state:
            self.h = h
        return h[-1]

    # assuming the loss function is cross-entropy
    def bprop(self, y):
        di = y
        nlayers = len(self.layers)
        for i in reversed(range(nlayers)):
            layer = self.layers[i]
            di = layer.backward(di, self.h[i+1], self.h[i])
        return