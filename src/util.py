import numpy
from scipy.stats import entropy
from src.network import NeuralNetwork

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


class TrainSettings(object):
    def __init__(self, learning_rate=0.01, momentum=0.0, l2=0.0, l1=0.0, dropout=0.0,
                 epoch=200, batch_size=64, auto_stop=False, auto_plot=False, auto_visualize=False,
                 plot_callback=None, loss_callback=None, filename=None):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2 = l2
        self.l1 = l1
        self.dropout = dropout
        self.epoch = epoch
        self.batch_size = batch_size
        self.auto_stop = auto_stop
        self.auto_plot = auto_plot
        self.auto_visualize = auto_visualize
        self.plot_callback = plot_callback
        self.loss_callback = loss_callback
        self.filename = filename


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
            x1 = self.x_shuffled[self.draw_num:self.size]
            y1 = self.y_shuffled[self.draw_num:self.size]
            self.shuffle()
            x2 = self.x_shuffled[0:leftover]
            y2 = self.y_shuffled[0:leftover]
            self.draw_num = leftover
            x = numpy.append(x1, x2, axis=0)
            y = numpy.append(y1, y2, axis=0)
            return True, x, y
        elif leftover == 0:
            x = self.x_shuffled
            y = self.y_shuffled
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
            x1 = self.x_shuffled[self.draw_num:self.size]
            self.shuffle()
            x2 = self.x_shuffled[0:leftover]
            self.draw_num = leftover
            x = numpy.append(x1, x2, axis=0)
            return True, x
        elif leftover == 0:
            x = self.x_shuffled
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


class Status(object):
    def __init__(self, train_settings: TrainSettings, data_store: DataStore, is_train: bool):
        self.target_epoch = train_settings.epoch
        self.current_epoch = 0
        self.error: numpy.ndarray = numpy.zeros(shape=(self.target_epoch,), dtype=numpy.float32)
        self.loss: numpy.ndarray = numpy.zeros(shape=(self.target_epoch,), dtype=numpy.float32)
        self.perplexity: numpy.ndarray = None
        self.soft_prob: numpy.ndarray = None
        self.predict: numpy.ndarray = None
        self.x_batch: numpy.ndarray = None
        self.y_batch: numpy.ndarray = None
        self.size: int = data_store.size
        self.is_train: bool = is_train
        self.train_settings: TrainSettings = train_settings
        self.data_store: DataStore = data_store
        if self.is_train:
            self.batch_size = self.train_settings.batch_size
            self.draw_batch = self.draw_batch_train
        else:
            self.batch_size = self.size
            self.draw_batch = self.draw_direct

    def draw_batch_train(self):
        return self.data_store.draw_batch_dual(self.batch_size)

    def draw_direct(self):
        return self.data_store.draw_direct_dual()



def print_log(status_train: Status, status_valid: Status):
    current_epoch = status_train.current_epoch
    print('\t', status_train.current_epoch, '\t', sep='', end=' ')
    print('\t|\t ', "{0:.5f}".format(status_train.loss[current_epoch]),
          '  \t|\t ', "{0:.5f}".format(status_train.error[current_epoch]),
          '  \t|\t ', "{0:.5f}".format(status_valid.loss[current_epoch]),
          '  \t|\t ', "{0:.5f}".format(status_valid.error[current_epoch]),
          '\t',
          sep='')


def iteration(nn: NeuralNetwork, status: Status):
    increase_epoch, status.x_batch, status.y_batch = status.draw_batch()
    status.soft_prob = nn.fprop(status.x_batch, status.is_train)
    status.predict = pick_class(status.soft_prob)
    status.train_settings.loss_callback(status)
    if status.is_train:
        nn.bprop(status.y_batch)
        nn.update(status.train_settings)
    return increase_epoch


def cross_train(nn: NeuralNetwork, data_store_train: DataStore, data_store_valid: DataStore,  train_settings: TrainSettings):
    print('------------------ Start Training -----------------')
    print('\tepoch\t|\ttrain loss\t|\ttrain error\t|\tvalid loss\t|\tvalid error\t')
    status_train = Status(train_settings, data_store_train, True)
    status_valid = Status(train_settings, data_store_valid, False)

    while status_train.current_epoch < status_train.target_epoch:
        if iteration(nn, status_train):
            iteration(nn, status_valid)
            print_log(status_train, status_valid)
            if status_train.current_epoch % 2 == 1:
                train_settings.plot_callback(status_train, status_valid)
            status_train.current_epoch += 1
            status_valid.current_epoch += 1
    return









