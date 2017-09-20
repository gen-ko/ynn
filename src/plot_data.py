import numpy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def reshape_row_major(x, num_row, num_column):
    # type: (numpy.ndarray, int, int) -> numpy.ndarray
    return numpy.reshape(x, (num_row, num_column))

def plot_image(x_reshaped):

    return plt.imshow(x_reshaped, cmap='gray')

#def plot_error()