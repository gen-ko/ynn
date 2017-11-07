import numpy


# load 1 dimension / flatten data with the label as the last element of each line
def load_from_path(data_filepath):
    # data_train: numpy.ndarray
    data_input = numpy.loadtxt(data_filepath, delimiter=',')
    # x_train: numpy.ndarray
    x_dimension = data_input.shape[1] - 1
    x = data_input[:, 0: x_dimension]
    y = data_input[:, x_dimension].astype(int)
    return x, y


class DataStore(object):
    def __init__(self):
        self.hasData = False
        self.dataNum = 0
        self.dataDim = 0
        self.dataPath: str = ''

    # def feed_data(self, path: str):: TODO