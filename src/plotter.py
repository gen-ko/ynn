import numpy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

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


# TODO: A plloter for RBM, NN, AutoEncoder needs to be implemented
