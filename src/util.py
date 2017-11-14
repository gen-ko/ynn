import numpy
from scipy.stats import entropy

def cross_entropy_loss(predict_prob: numpy.ndarray, y_label: numpy.ndarray, take_average: bool = False) -> float:
    loss = 0.0
    batch_size = predict_prob.shape[0]
    for i in range(y_label.size):
        try:
            loss += -numpy.log(predict_prob[i, y_label[i]])
        except:
            raise ValueError('Cross Entropy Loss Overflowing')
    if take_average:
        loss /= batch_size
    return loss


def pick_class(softprob: numpy.ndarray) -> numpy.ndarray:
    classes = numpy.argmax(softprob, axis=1)
    return classes


def shuffle(x, y) -> (numpy.ndarray, numpy.ndarray):
    shuffle_idx = numpy.arange(y.size)
    numpy.random.shuffle(shuffle_idx)
    x = x[shuffle_idx]
    y = y[shuffle_idx]
    return x, y


def predict_score(predict_label: numpy.ndarray, y_label: numpy.ndarray, take_average: bool = False) -> float:
    tmp: float = numpy.sum(predict_label == y_label).astype(float)
    if take_average:
        tmp /= y_label.size
    return tmp


def perplexity(prob_distribution: numpy.ndarray) -> float:
    tmp = entropy(prob_distribution.T)
    tmp = numpy.power(2, tmp)
    return numpy.sum(tmp)





class DataStore(object):
    def __init__(self, x: numpy.ndarray, y: numpy.ndarray=None):
        self.x = x
        self.size = x.shape[0]
        self.draw_num = 0
        self.x_shuffled: numpy.ndarray = None
        self.y_shuffled: numpy.ndarray = None
        try:
            self.y = y
            self.y_exists = True
            self.draw_batch = self.draw_batch_dual
            self.draw_direct = self.draw_direct_dual
            self.shuffle = self.shuffle_dual
            self.shuffle()
        except TypeError:
            self.y = None
            self.y_exists = False
            self.draw_batch = self.draw_batch_x
            self.draw_direct = self.draw_direct_x
            self.shuffle = self.shuffle_single
            self.shuffle()
        return

    def shuffle_dual(self):
        shuffle_idx = numpy.arange(self.size)
        numpy.random.shuffle(shuffle_idx)
        self.x_shuffled = self.x[shuffle_idx]
        self.y_shuffled = self.y[shuffle_idx]
        return

    def shuffle_single(self):
        shuffle_idx = numpy.arange(self.size)
        numpy.random.shuffle(shuffle_idx)
        self.x_shuffled = self.x[shuffle_idx]
        return

    def draw_batch_dual(self, batch_size: int) -> (bool, numpy.ndarray, numpy.ndarray):
        leftover: int = batch_size + self.draw_num - self.size
        if leftover > 0:
            self.shuffle()
            x1 = self.x_shuffled[self.draw_num:self.size]
            y1 = self.y_shuffled[self.draw_num:self.size]

            x2 = self.x_shuffled[0:leftover]
            y2 = self.y_shuffled[0:leftover]
            self.draw_num = leftover
            x = numpy.append(x1, x2, axis=0)
            y = numpy.append(y1, y2, axis=0)
            return True, x, y
        elif leftover == 0:
            x = self.x_shuffled[self.draw_num:self.size]
            y = self.y_shuffled[self.draw_num:self.size]
            self.shuffle()
            self.draw_num = 0
            return True, x, y
        else:
            next_draw_num = self.draw_num + batch_size
            x = self.x_shuffled[self.draw_num:next_draw_num]
            y = self.y_shuffled[self.draw_num:next_draw_num]
            self.draw_num = next_draw_num
            return False, x, y

    def draw_batch_x(self, batch_size: int) -> (bool, numpy.ndarray):
        leftover: int = batch_size + self.draw_num - self.size
        if leftover > 0:
            self.shuffle()
            x1 = self.x_shuffled[self.draw_num:self.size]

            x2 = self.x_shuffled[0:leftover]
            self.draw_num = leftover
            x = numpy.append(x1, x2, axis=0)
            return True, x
        elif leftover == 0:
            x = self.x_shuffled[self.draw_num:self.size]
            self.shuffle()
            self.draw_num = 0
            return True, x
        else:
            next_draw_num = self.draw_num + batch_size
            x = self.x_shuffled[self.draw_num:next_draw_num]
            self.draw_num = next_draw_num
            return False, x

    def draw_direct_dual(self, *args) -> (bool, numpy.ndarray, numpy.ndarray):
        return True, self.x, self.y

    def draw_direct_x(self, *args) -> (bool, numpy.ndarray):
        return True, self.x


class TrainSettings(object):
    def __init__(self, learning_rate=0.01, momentum=0.0, l2=0.0, l1=0.0, dropout=0.0,
                 epoch=200, batch_size=64, auto_stop=False, auto_plot=False, auto_visualize=False,
                 plot_callback=None, loss_callback=None, filename='f', prefix='p', infix='i', suffix='s'):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2: float = l2
        self.l1: float = l1
        self.dropout = dropout
        self.epoch = epoch
        self.batch_size = batch_size
        self.auto_stop = auto_stop
        self.auto_plot = auto_plot
        self.auto_visualize = auto_visualize
        self.plot_callback = plot_callback
        self.loss_callback = loss_callback
        self.filename = filename
        self.prefix = prefix
        self.infix = infix
        self.suffix = suffix
        return











