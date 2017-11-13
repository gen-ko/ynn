from src.network import NeuralNetwork
import src.util as uf
import numpy
from src.util import DataStore
from src.util import TrainSettings
import os
import pickle


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
    status.predict = uf.pick_class(status.soft_prob)
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
            if status_train.current_epoch % 20 == 19:
                train_settings.plot_callback(status_train, status_valid)
            status_train.current_epoch += 1
            status_valid.current_epoch += 1

    filename = train_settings.filename + '-' + train_settings.prefix + '-' + train_settings.infix + '-' + train_settings.suffix + '.dump'
    full_path = os.path.realpath(__file__)
    path, _ = os.path.split(full_path)
    savepath = os.path.join(path, '../output/dump', filename)

    with open(savepath, 'wb+') as f:
        pickle.dump(nn.dump(), f)
    return


def inference(nn: NeuralNetwork, data_store: DataStore, settings: TrainSettings):
    status = Status(settings, data_store, False)
    iteration(nn, status)
    return status.predict


def nlp_inference_sentence(nn: NeuralNetwork, head: list, vocab: dict):
    inv_vocab = {v: k for k, v in vocab.items()}
    if len(head) is not 3:
        raise ValueError(f'the length of the sentence head should be 3, but is actually {len(head)}')
    word_next = 'START'
    while word_next != 'END' and len(head) <= 13:
        id1 = vocab[head[-3]]
        id2 = vocab[head[-2]]
        id3 = vocab[head[-1]]
        data_input = numpy.array([[id1, id2, id3]])
        id_prob = nn.fprop(data_input)
        id_next = uf.pick_class(id_prob)[0]
        word_next = inv_vocab[id_next]
        head.append(word_next)
    return head





