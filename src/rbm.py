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


class RBM(object):
    def __init__(self, input_units, hidden_units):
        self.layer = layer.RBM(input_dimension=input_units, output_dimension=hidden_units)
        self.is_autostop_enabled = False
        self.is_plot_enabled = False
        self.is_visualize_enabled = False
        self.is_dump_enabled = False


    def gibbs_sampling(self, x_t_sample):
        # x shape (sample_dimension, 1)
        h_t_prob = self.layer.forward(x_t_sample)
        h_t_sample = self.sample_h(h_t_prob)
        h_neg_prob = numpy.array(h_t_prob)
        h_neg_sample = numpy.array(h_t_sample)
        for ki in range(0, self.k, 1):
            x_neg_prob = self.layer.backward(h_neg_sample)
            x_neg_sample = self.sample_x(x_neg_prob)
            h_neg_prob = self.layer.forward(x_neg_sample)
            h_neg_sample = self.sample_h(h_neg_prob)
        return [x_neg_prob, x_neg_sample, h_t_prob, h_neg_prob, h_neg_sample]

    def reconstruct(self, x_t_sample):
        # x shape (sample_dimension, 1)
        h_t_prob = self.layer.forward(x_t_sample)
        x_reconstruct = self.layer.backward(h_t_prob)
        return x_reconstruct

    @staticmethod
    def sample_h(h_p):
        h_sample = numpy.random.binomial(n=1, p=h_p, size=h_p.shape)
        return h_sample

    @staticmethod
    def sample_x(x_p):
        x_sample = numpy.random.binomial(n=1, p=x_p, size=x_p.shape)
        return x_sample

    def update(self, x_t, x_neg_prob, h_t_prob, h_neg_prob):
        delta_w = numpy.zeros(shape=self.layer.w.shape, dtype=numpy.float64)
        delta_h_bias = numpy.zeros(shape=self.layer.h_bias.shape)
        delta_x_bias = numpy.zeros(shape=self.layer.x_bias.shape)
        batch_size = x_t.shape[1]
        for i in range(0, batch_size, 1):
            delta_w += numpy.outer(h_t_prob[:, i], x_t[:, i]) - numpy.outer(h_neg_prob[:, i], x_neg_prob[:, i])
            tmp1 = h_t_prob[:, i]
            tmp2 = h_neg_prob[:, i]
            delta_h_bias += (tmp1 - tmp2).reshape(delta_h_bias.shape[0], delta_h_bias.shape[1])
            delta_x_bias += (x_t[:, i] - x_neg_prob[:, i]).reshape(delta_x_bias.shape)

        delta_w /= batch_size
        delta_h_bias /= batch_size
        delta_x_bias /= batch_size

        self.layer.update(delta_w=delta_w,
                          delta_h_bias=delta_h_bias,
                          delta_x_bias=delta_x_bias,
                          learning_rate=self.learning_rate)

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

    def train(self, x_train, x_valid, k=1, epoch=200, learning_rate=0.01, batch_size=128, dump=False, plotfile='time'):
        self._dump = dump
        print('------------------ Start Training -----------------')
        print('------------- using cross-entropy loss ------------')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.plot_loss_train = numpy.zeros((epoch, ), dtype=numpy.float64)
        self.plot_loss_valid = numpy.zeros((epoch, ), dtype=numpy.float64)
        self.k = k

        for j in range(epoch):
            x_train = self.shuffle(x_train)
            loss_train = 0.0
            if j % 20 == 0:
                print('|\tepoch\t|\ttrain loss\t|\tvalid loss\t|')
            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train[i: i + batch_size]
                x_batch = x_batch.T
                x_t_sample = x_batch
                x_neg_prob, x_neg_sample, h_t_prob, h_neg_prob, h_neg_sample = self.gibbs_sampling(x_t_sample=x_t_sample)
                x_reconstruct = self.reconstruct(x_t_sample)
                loss_train += self.cross_entropy_loss(x_t_sample, x_reconstruct, False)
                self.update(x_t_sample, x_neg_sample, h_t_prob, h_neg_prob)
            x_valid_sample = x_valid.T
            x_valid_neg_prob = self.reconstruct(x_valid_sample)
            loss_train /= x_train.shape[0]
            loss_valid = self.cross_entropy_loss(x_valid_sample, x_valid_neg_prob, False)
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
        self.autostop(current_epoch=3000, plotfile=plotfile)
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
        output_prob, output, _, _, _ = self.gibbs_sampling(x_input)
        return output_prob


    def dump(self, current_epoch, plotfile):
        # fix the absolute path of file
        full_path = os.path.realpath(__file__)
        path, _ = os.path.split(full_path)
        filename_prefix = 'rbm-'
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
            tmp = -numpy.multiply(x_sample, numpy.log(x_reconstruct)) -numpy.multiply((1 - x_sample),
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
        filename_prefix = 'rbm-'
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
        filename_prefix = 'rbm-'
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
