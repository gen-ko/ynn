import pickle
import os
import matplotlib.pyplot as plt

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
data_filepath = '../temp'
network_dump_filename = 'network.dump'
train_error_dump_filename = 'train_error.dump'
valid_error_dump_filename = 'valid_error.dump'

network_dump_filepath = os.path.join(path, data_filepath, network_dump_filename)
train_error_dump_filepath = os.path.join(path, data_filepath, train_error_dump_filename)
valid_error_dump_filepath = os.path.join(path, data_filepath, valid_error_dump_filename)

with open(network_dump_filepath, 'rb') as f:
    nn = pickle.load(f)

with open(train_error_dump_filepath, 'rb') as f:
    train_error = pickle.load(f)

with open(valid_error_dump_filepath, 'rb') as f:
    valid_error = pickle.load(f)




line_1, = plt.plot(nn._loss, label='train loss')
line_2, = plt.plot(nn._loss_valid, label='valid loss')
plt.legend(handles=[line_1, line_2])
plt.ylim(0, 0.2)
plt.xlim(0,200)
plt.xlabel('epoch')
plt.ylabel('cross-entropy loss')
plt.savefig(os.path.join(path, '../output/output-fig1.png'))

plt.figure()
line_1, = plt.plot(train_error, label='train error')
line_2, = plt.plot(valid_error, label='valid error')
plt.legend(handles=[line_1, line_2])
plt.xlabel('epoch')
plt.ylabel('error rate')
plt.savefig(os.path.join(path, '../output/output-fig2.png'))

print('hi-2')
