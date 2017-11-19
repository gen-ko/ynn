import numpy

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
        return self.data_store.draw_batch(self.batch_size)

    def draw_direct(self):
        return self.data_store.draw_direct()





