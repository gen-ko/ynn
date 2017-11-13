import src.train as utf
import src.util as uf
import numpy
import src.plotter as plotter
import src.train as utf


def loss_callback(status: utf.Status):
    current_epoch = status.current_epoch

    status.loss[current_epoch] += (uf.cross_entropy_loss(status.soft_prob,
                                                         status.y_batch,
                                                         take_average=False) / status.size)
    status.error[current_epoch] += (numpy.sum(status.predict != status.y_batch) / status.size)
    try:
        status.perplexity[current_epoch] += uf.perplexity(status.soft_prob) / status.size
    except TypeError:
        status.perplexity = numpy.zeros(shape=(status.target_epoch,), dtype=numpy.float32)
        tmp = uf.perplexity(status.soft_prob) / status.size
        status.perplexity[current_epoch] += tmp
    return

def plot_callback(status_train: utf.Status, status_valid: utf.Status):
    plotter.plot_loss(status_train, status_valid)
    plotter.plot_perplexity(status_train, status_valid)
    return