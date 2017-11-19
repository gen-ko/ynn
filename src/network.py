
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
'''
class NeuralNetwork_Dumpable(object):
    def __init__(self, layers: [layer.Layer], learning_rate=0.01, regularizer=0.0001, debug=False,
                 momentum=0.9):

        self._num_layer = len(layers)
        self.layers = layers
        self._learning_rate = learning_rate
        self._regularizer = regularizer
        self._H = [numpy.zeros((layers[0].input_dimension, ), dtype=numpy.float32)]
        self._H += [[numpy.zeros((layer.output_dimension, ), dtype=numpy.float32)] for layer in self.layers]
        self._num_class = layers[-1].output_dimension
        self._debug = debug
        self.epoch = 200
        self._momentum = momentum
        self.g_h: list
        self.train_error = numpy.zeros(shape=(1,), dtype=numpy.float32)
        self.valid_error = numpy.zeros(shape=(1,), dtype=numpy.float32)
        self.train_loss = numpy.zeros(shape=(1,), dtype=numpy.float32)
        self.valid_loss = numpy.zeros(shape=(1,), dtype=numpy.float32)


    def fprop(self, x):
        self._H[0] = x
        for i in range(0, self._num_layer, 1):
            self._H[i + 1] = self.layers[i].forward(self._H[i])

    def bprop(self, y):
        g_h = y
        for i in range(self._num_layer - 1, -1, -1):
            g_h = self.layers[i].backward(g_h, h_out=self._H[i + 1], h_in=self._H[i])

    def update(self):
        for layer in self.layers:
            layer.update(self._learning_rate, self._regularizer, self._momentum)


    def train(self, x_train, y_train, x_valid, y_valid, epoch, batch_size=128, dump=False):
        self._dump = dump
        print('------------------ Start Training -----------------')
        print('\tepoch\t|\ttrain loss\t|\ttrain error\t|\tvalid loss\t|\tvalid error\t')
        self.train_error = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        self.valid_error = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        self.train_loss = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        self.valid_loss = numpy.zeros(shape=(epoch,), dtype=numpy.float64)
        self.batch_size = batch_size
        for j in range(epoch):
            x_train, y_train = uf.shuffle(x_train, y_train)
            train_score = 0.0
            for i in range(0, y_train.size, batch_size):
                x_batch = x_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]
                self.fprop(x_batch)
                self.train_loss[j] += uf.cross_entropy_loss(self._H[-1], y_batch, take_average=False)
                y_predict = uf.pick_class(self._H[-1])
                train_score += uf.predict_score(y_predict, y_batch, take_average=False)
                self.bprop(y_batch)
                self.update()
            # start a validation
            self.train_loss[j] = self.train_loss[j] / y_train.shape[0]
            self.train_error[j] = (1.0 - train_score / y_train.shape[0])
            self.valid_pass(x_valid, y_valid, j)


            print('\t', j, '\t', sep='', end=' ')
            print('\t|\t ', "{0:.5f}".format(self.train_loss[j]),
                  '  \t|\t ', "{0:.5f}".format(self.train_error[j]),
                  '  \t|\t ', "{0:.5f}".format(self.valid_loss[j]),
                  '  \t|\t ', "{0:.5f}".format(self.valid_error[j]),
                  '\t',
                  sep='')
        return


    def valid_pass(self, x_valid, y_valid, epoch):
        self.fprop(x_valid)
        self.valid_loss[epoch] = uf.cross_entropy_loss(self._H[-1], y_valid)
        y_predict = uf.pick_class(self._H[-1])
        valid_score = numpy.sum(y_predict == y_valid)
        self.valid_loss[epoch] /= y_valid.shape[0]
        self.valid_error[epoch] = 1.0 - valid_score / y_valid.shape[0]
'''


