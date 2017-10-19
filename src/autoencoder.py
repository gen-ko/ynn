import numpy
import pickle
from src import layer
import os
from time import gmtime, strftime
import matplotlib.pyplot as plt



# initialize the RBM model constructor before using
def init_rbm(random_seed=1099):
    # set a random seed to ensure the consistence between different runs
    numpy.random.seed(1099)
    return


class AutoEncoder(object):
    def __init__(self, input_units, hidden_units):
        self.layer = layer.AutoEncoder(input_dimension=input_units, output_dimension=hidden_units)
        self.is_autostop_enabled = False
        self.is_plot_enabled = False
        self.is_visualize_enabled = False
        self.is_dump_enabled = False


    def reconstruct(self, x):
        # x shape (sample_dimension, 1)

        h = self.layer.forward(x)
        x_neg = self.layer.backward(h)
        return [x_neg, h]

    def update(self, x, x_neg, h):
        d_a2 = numpy.multiply(1-x, x_neg) - numpy.multiply(x, 1-x_neg)
        d_w2  = numpy.zeros(shape=self.layer.w.T.shape, dtype=numpy.float64)
        d_x_bias = numpy.zeros(shape=(x.shape[0], 1), dtype=numpy.float64)

        for i in range(d_a2.shape[1]):
            d_w2 += numpy.outer(h[:, i], d_a2[:, i]).T
            d_x_bias += d_a2[:, i].reshape(self.layer.x_bias.shape)

        d_w2 /= d_a2.shape[1]
        d_x_bias /= d_a2.shape[1]

        d_h = numpy.dot(self.layer.w, d_a2)

        d_a1 = numpy.multiply(d_h, numpy.multiply(h, 1-h))
        d_w1 = numpy.zeros(shape=self.layer.w.shape, dtype=numpy.float64)
        d_h_bias = numpy.zeros(shape=(h.shape[0], 1), dtype=numpy.float64)

        for i in range(d_a1.shape[1]):
            d_w1 += numpy.outer(d_a1[:, i], x[:, i])
            d_h_bias += d_a1[:, i].reshape(self.layer.h_bias.shape)

        d_w1 /= d_a1.shape[1]
        d_h_bias /= d_a1.shape[1]

        d_w = d_w1 + d_w2.T

        self.layer.update(delta_w=-d_w,
                          delta_x_bias=-d_x_bias,
                          delta_h_bias=-d_h_bias,
                          learning_rate=self.learning_rate)
        return


    def shuffle(self, x):
        shuffle_idx = numpy.arange(x.shape[0])
        numpy.random.shuffle(shuffle_idx)
        x = x[shuffle_idx]
        return x

    def set_visualize(self, dim1, dim2, stride=20, enabled=False):
        self.visualize_x_dim = dim1
        self.visualize_y_dim = dim2
        self.is_visualize_enabled = enabled
        self.visualize_stride = stride

    def set_plot(self, stride=20, enabled=False):
        self.is_plot_enabled = enabled
        self.plot_stride = stride

    def set_dump(self, stride=20, enabled=False):
        self.is_dump_enabled = enabled
        self.dump_stride = stride

    def set_autostop(self, window=20, stride=20):
        self.is_autostop_enabled = True
        self.stop_stride = stride
        self.stop_window = window

    def convergence_detection(self, current_epoch):
        if current_epoch > 2 * self.stop_window:

            loss_last = numpy.mean(self.plot_loss_valid[current_epoch-self.stop_window+1:current_epoch+1])
            loss_recent = numpy.mean(self.plot_loss_valid[current_epoch-2*self.stop_window+1:current_epoch-self.stop_window+1])
            if loss_last > loss_recent:
                return True
        return False

    def train(self, x_train, x_valid, k=1, epoch=200,
              learning_rate=0.01, batch_size=128,
              dump=False, plotfile='time',
              dropout=False, dropout_rate=0.1):
        self._dump = dump
        print('------------------ Start Training -----------------')
        print('------------- using cross-entropy loss ------------')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.plot_loss_train = numpy.zeros((epoch, ), dtype=numpy.float64)
        self.plot_loss_valid = numpy.zeros((epoch, ), dtype=numpy.float64)
        self.k = k
        self.dropout=dropout
        self.dropout_rate=dropout_rate
        for j in range(epoch):
            x_train = self.shuffle(x_train)
            loss_train = 0.0
            if j % 20 == 0:
                print('|\tepoch\t|\ttrain loss\t|\tvalid loss\t|')
            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train[i: i + batch_size]
                x_batch = x_batch.T

                y = numpy.array(x_batch)
                if self.dropout:
                    for ii in range(x_batch.shape[0]):
                        for jj in range(x_batch.shape[1]):
                            if numpy.random.uniform(0, 1, ) < self.dropout_rate:
                                y[ii, jj] = 0.0

                x_neg,  h = self.reconstruct(x=y)
                loss_train += self.cross_entropy_loss(x_batch, x_neg, False)
                self.update(x_batch, x_neg, h)
            x_valid_sample = x_valid.T
            x_valid_neg, _ = self.reconstruct(x=x_valid_sample)
            loss_train /= x_train.shape[0]
            loss_valid = self.cross_entropy_loss(x_valid_sample, x_valid_neg, False)
            loss_valid /= x_valid.shape[0]

            print('\t', j, '\t', sep='', end=' ')
            print('\t|\t ', "{0:.5f}".format(loss_train),
                  '  \t|\t ', "{0:.5f}".format(loss_valid),
                  '\t',
                  sep='')
            self.plot_loss_train[j] = loss_train
            self.plot_loss_valid[j] = loss_valid
            if self.is_visualize_enabled:
                if j % self.visualize_stride == self.visualize_stride - 1:
                    self.visualize(current_epoch=j, plotfile=plotfile)
            if self.is_plot_enabled:
                if j % self.plot_stride == self.plot_stride - 1:
                    self.plot(current_epoch=j, plotfile=plotfile)
            if self.is_dump_enabled:
                if j % self.dump_stride == self.dump_stride - 1:
                    self.dump(current_epoch=j, plotfile=plotfile)
            if self.is_autostop_enabled:
                if j % self.stop_stride == self.stop_stride - 1:
                    if self.convergence_detection(j):
                        self.autostop(current_epoch=j, plotfile=plotfile)
                        print('---- Earling stopping now ----')
                        return
        self.autostop(current_epoch=epoch, plotfile=plotfile)
        return

    def autostop(self, current_epoch, plotfile):
        plotfile += '-autostop'
        self.visualize(current_epoch=current_epoch, plotfile=plotfile)
        self.plot(current_epoch=current_epoch, plotfile=plotfile)
        self.dump(current_epoch=current_epoch, plotfile=plotfile)
        return

    def generate_sample(self, sample_num=1, sampler_step=1000):
        self.k = sampler_step
        # generate an input
        x_input = numpy.random.binomial(n=1, p=0.5, size=(self.layer.x_bias.shape[0], sample_num))
        output,  _ = self.reconstruct(x_input)
        return output


    def dump(self, current_epoch, plotfile):
        # fix the absolute path of file
        full_path = os.path.realpath(__file__)
        path, _ = os.path.split(full_path)
        filename_prefix = 'AE-'
        filename_infix = strftime("%y%m%d%H%M%S", gmtime()) + '-'
        filename_suffix = 'whx-'
        filename_epoch = str(current_epoch)
        filename_extension = '.dump'

        if plotfile == 'time':
            filename = filename_prefix + filename_infix + filename_suffix + filename_epoch + filename_extension
        else:
            filename = plotfile + '-' + filename_prefix + filename_suffix + filename_epoch + filename_extension

        file_path = os.path.join(path, '../output/dump', filename)
        with open(file_path, 'wb') as f:
            pickle.dump([self.layer.w, self.layer.h_bias, self.layer.x_bias], f)
        return

    def cross_entropy_loss(self, x_sample, x_reconstruct, take_average=True):
        loss = 0.0
        # the loss between input x and the reconstruction of x
        # x may be a batch (sample_dimension, batch_size)
        try:
            tmp = -numpy.multiply(x_sample, numpy.log(x_reconstruct)) -numpy.multiply(1 - x_sample,
                                                                                      numpy.log(1 - x_reconstruct))
            if take_average:
                loss = sum(sum(tmp)) / x_sample.shape[1]
            else:
                loss = sum(sum(tmp))
        except:
            print('error, loss not computable')
        return loss

    def visualize(self, current_epoch, plotfile):
        # fix the absolute path of file
        full_path = os.path.realpath(__file__)
        path, _ = os.path.split(full_path)
        filename_prefix = 'AE-'
        filename_infix = strftime("%y%m%d%H%M%S", gmtime()) + '-'
        filename_suffix = 'visualize-'
        filename_epoch = str(current_epoch)
        filename_extension = '.png'

        if plotfile == 'time':
            filename = filename_prefix + filename_infix + filename_suffix + filename_epoch + filename_extension
        else:
            filename = plotfile + '-' + filename_prefix + filename_suffix + filename_epoch + filename_extension

        figure_path = os.path.join(path, '../output/visualize', filename)

        ncol = 10
        nrow = int(self.layer.w.shape[0] / ncol)
        plt.figure(2)
        plt.axis('off')
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        for i in range(self.layer.w.shape[0]):
            ax = plt.subplot(nrow, ncol, i + 1)
            ax.imshow(self.layer.w[i, :].reshape([self.visualize_x_dim, self.visualize_y_dim]))
            ax.axis('off')
        plt.savefig(os.path.join(path, figure_path))
        plt.close(2)
        return

    def plot(self, current_epoch, plotfile):
        full_path = os.path.realpath(__file__)
        path, _ = os.path.split(full_path)
        filename_prefix = 'AE-'
        filename_infix = strftime("%y%m%d%H%M%S", gmtime()) + '-'
        filename_suffix = 'cost'
        filename_extension = '.png'
        if plotfile == 'time':
            filename = filename_prefix + filename_infix + filename_suffix + filename_extension
        else:
            filename = plotfile + '-' + filename_prefix + filename_suffix + filename_extension
        titletext = f'lr={self.learning_rate}, ' \
                    f'k={self.k},hidden units={self.layer._output_dimension},batch={self.batch_size}'

        plt.figure(1)
        line_1, = plt.plot(self.plot_loss_train[0:current_epoch], label='train loss')
        line_2, = plt.plot(self.plot_loss_valid[0:current_epoch], label='valid loss')
        plt.legend(handles=[line_1, line_2])
        plt.xlabel('epoch')
        plt.ylabel('cross-entropy cost')
        plt.title(titletext)
        plt.savefig(os.path.join(path, '../output/plot-loss', filename))
        plt.close(1)
        return
