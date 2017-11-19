import src.train as utf
import src.util as uf
import numpy
import src.plotter as plotter
import src.train as utf
from src.util import status
from src.util import metrics


def loss_callback(status: status.Status):
    current_epoch = status.current_epoch

    tmp_cross_entropy = (metrics.cross_entropy_loss(status.soft_prob, status.y_batch) / status.size)

    status.loss[current_epoch] += tmp_cross_entropy

    status.error[current_epoch] += (numpy.sum(status.predict != status.y_batch) / status.size)
    try:
        perp, weight = metrics.perplexity(status.soft_prob, status.y_batch)
        tmp = status.perplexity[current_epoch]
        tmp = tmp + perp * weight / status.size
        status.perplexity[current_epoch] = tmp
    except TypeError:
        status.perplexity = numpy.zeros(shape=(status.target_epoch,), dtype=numpy.float32)
        perp, weight = metrics.perplexity(status.soft_prob, status.y_batch)
        tmp = perp * weight / status.size
        status.perplexity[current_epoch] = tmp
    return

def plot_callback(status_train: status.Status, status_valid: status.Status):
    plotter.plot_loss(status_train, status_valid)
    plotter.plot_perplexity(status_train, status_valid)
    return