# required python version: 3.6+

import os
import sys
import src.load_data as load_data
from src import plot_data
from src import layer
import src.rbm as rbm
import matplotlib.pyplot as plt
import numpy
import os
import pickle

#  format of data
# disitstrain.txt contains 3000 lines, each line 785 numbers, comma delimited



print('start initializing...')
full_path = os.path.realpath(__file__)
path, _ = os.path.split(full_path)
data_filepath = '../output/dump'

filepath = os.path.join(path, data_filepath, 'script-2-2-k=5-autostop-rbm-whx-2359.dump')

# w (100, 784)
# h_bias (100, 1)
# x_bias (784, 1)
with open(filepath, 'rb') as f:
    w, h_bias, x_bias = pickle.load(f)


numpy.random.seed(1099)

myRBM = rbm.RBM(28*28, 100)
myRBM.layer.w = w
myRBM.layer.h_bias = h_bias
myRBM.layer.x_bias = x_bias

x = myRBM.generate_sample(sample_num=100, sampler_step=1000)

filename_prefix = 'rbm-generated-'
filename_suffix = 'visualize-2359'
filename_extension = '.png'


filename = filename_prefix + filename_suffix + filename_extension

figure_path = os.path.join(path, '../output/generate', filename)

ncol = 10
nrow = int(x.shape[1] / ncol)
plt.figure(3)
plt.axis('off')
plt.subplots_adjust(wspace=0.01, hspace=0.01)
for i in range(x.shape[1]):
    ax = plt.subplot(nrow, ncol, i + 1)
    ax.imshow(x[:, i].reshape(28,28))
    ax.axis('off')
plt.savefig(os.path.join(path, figure_path))
plt.close(3)