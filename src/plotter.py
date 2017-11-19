import numpy
import matplotlib.pyplot as plt
import os
from src import util as uf
from src import train as utf

def reshape_row_major(x, num_row, num_column):
    # type: (numpy.ndarray, int, int) -> numpy.ndarray
    return numpy.reshape(x, (num_row, num_column))

def plot_image(x_reshaped):

    return plt.imshow(x_reshaped, cmap='gray')


class PlotterBase(object):
    def __init__(self, plotfile, filename_prefix, filename_suffix):
        self.plotfile = plotfile
        filename_extension = '.png'
        self.filename = plotfile + '-' + filename_prefix + filename_suffix + filename_extension
        full_path = os.path.realpath(__file__)
        path, _ = os.path.split(full_path)
        self.savepath = os.path.join()

    def plot(self, *args):
        raise ValueError('Calling a virtual function')


class PlotterLoss(PlotterBase):
    def __init__(self, plotfile, filename_prefix, filename_suffix):
        PlotterBase.__init__(self, plotfile, filename_prefix, filename_suffix)

    def set_title(self, learning_rate, hidden_units, batch_size, k):
        self.titletext = f'lr={learning_rate}, ' \
                    f'k={k},hidden units={hidden_units},batch={batch_size}'

    def plot(self, current_epoch, loss_train, loss_valid):
        plt.figure(1)
        line_1, = plt.plot(loss_train[0:current_epoch], label='train loss')
        line_2, = plt.plot(loss_valid[0:current_epoch], label='valid loss')
        plt.legend(handles=[line_1, line_2])
        plt.xlabel('epoch')
        plt.ylabel('cross-entropy cost')
        plt.title(self.titletext)
        plt.savefig(self.savepath)
        plt.close(1)
        return

def plot_base(v1, v2, label, save_path):
    plt.figure(1)
    label_1 = 'train' + ' ' + label
    label_2 = 'valid' + ' ' + label
    line_1, = plt.plot(v1, label=label_1)
    line_2, = plt.plot(v2, label=label_2)
    plt.legend(handles=[line_1, line_2])
    plt.xlabel('epoch')
    plt.ylabel(label)
    #plt.title(self.titletext)
    plt.savefig(save_path)
    plt.close(1)
    return


def plot_perplexity(status_train: utf.Status, status_valid: utf.Status):
    try:
        plot_file = status_train.train_settings.filename
    except ValueError:
        plot_file = 'loss'
    try:
        filename_prefix = status_train.train_settings.prefix
    except AttributeError:
        filename_prefix = 'prefix'
    try:
        filename_suffix = status_train.train_settings.suffix
    except AttributeError:
        filename_suffix = 'perp'
    filename_extension = '.png'
    filename = plot_file + '-' + filename_prefix + '-' + filename_suffix + '-perp' + filename_extension
    full_path = os.path.realpath(__file__)
    path, _ = os.path.split(full_path)
    savepath = os.path.join(path, '../output/perplexity', filename)

    v1 = status_train.perplexity[0:status_train.current_epoch+1]
    v2 = status_valid.perplexity[0:status_valid.current_epoch+1]
    label = 'perplexity'
    plot_base(v1, v2, label, savepath)
    return


def plot_loss(status_train: utf.Status, status_valid: utf.Status):
    try:
        plot_file = status_train.train_settings.filename
    except AttributeError:
        plot_file = 'loss'
    try:
        filename_prefix = status_train.train_settings.prefix
    except AttributeError:
        filename_prefix = 'prefix'
    try:
        filename_suffix = status_train.train_settings.suffix
    except AttributeError:
        filename_suffix = 'loss'
    filename_extension = '.png'
    filename = plot_file + '-' + filename_prefix + '-' + filename_suffix + '-loss' + filename_extension
    full_path = os.path.realpath(__file__)
    path, _ = os.path.split(full_path)
    savepath = os.path.join(path, '../output/plot-loss', filename)

    v1 = status_train.loss[0:status_train.current_epoch+1]
    v2 = status_valid.loss[0:status_valid.current_epoch+1]
    label = 'loss'
    plot_base(v1, v2, label, savepath)
    return





# TODO: A plloter for RBM, NN, AutoEncoder needs to be implemented
