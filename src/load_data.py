import numpy


def load_from_path(data_filepath):
    # data_train: numpy.ndarray
    data_input = numpy.loadtxt(data_filepath, delimiter=',')
    # x_train: numpy.ndarray
    x_dimension = data_input.shape[1] - 1
    x = data_input[:, 0: x_dimension]
    y = data_input[:, x_dimension].astype(int)
    return x, y

