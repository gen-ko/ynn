# This script is to visualize the parameters of the network

import numpy
import matplotlib.pyplot as plt
import pickle
import os

print('---------- starting script 1 - 3  ----------')

# fix the absolute path of file
full_path = os.path.realpath(__file__)
path, _ = os.path.split(full_path)
network_dump_path = os.path.join(path, '../temp/network.dump')

print('load network dump file...')
with open(network_dump_path, 'rb') as f:
    nn = pickle.load(f)


w: numpy.ndarray = nn.layers[0].w

ncol = 10
nrow = int(w.shape[1]/ncol)


fig = plt.figure()
print('ploting', end='')
plt.axis('off')
for i in range(w.shape[1]):
    ax = plt.subplot(nrow, ncol, i+1)
    im = ax.imshow(w[:, i].reshape([28, 28]))
    ax.axis('off')
    print('.', end='')
print('\n', end='')

plt.savefig(os.path.join(path, "../output/output-fig3.png"))