# required python version: 3.6+

import os
import sys
import src.load_data as load_data
from src import layer
import src.rbm as rbm
import src.autoencoder as autoencoder
import matplotlib.pyplot as plt
import numpy
import os
import pickle

# Phase 0: Read File
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
data_filepath = '../data'
save_filepath = '../output/n-gram'
dump_filepath = '../output/dump'
gram_name = '4-gram.png'
dictionary_name = 'dictionary.dump'
inv_dictionary_name = 'inv-dictionary.dump'
data_train_filename = 'train.txt'
data_valid_filename = 'val.txt'

data_train_filepath = os.path.join(path, data_filepath, data_train_filename)
data_valid_filepath = os.path.join(path, data_filepath, data_valid_filename)
gram_savepath = os.path.join(path, save_filepath, gram_name)
dictionary_dumppath = os.path.join(path, dump_filepath, dictionary_name)
inv_dictionary_dumppath = os.path.join(path, dump_filepath, inv_dictionary_name)

# load text data from path
def load_from_path(data_filepath):
    # data_train: numpy.ndarray
    data_input = numpy.loadtxt(data_filepath, dtype=str, delimiter='\n')
    # x_train: numpy.ndarray
    return data_input
data_train = load_from_path(data_train_filepath)
data_valid = load_from_path(data_valid_filepath)


# Phase 1: split

# Create a vocabulary dictionary you are required to create an entry for every word in the training set
# also, make the data lower-cased
vocabulary: dict = {}
all_lines: list = []
train_size = data_train.size
# will serve as a lookup table for the words and their corresponding id.
for i in range(train_size):
    tmp_list = data_train[i].lower().split()
    tmp_list.insert(0, 'START')
    tmp_list.append('END')
    all_lines.insert(i,  tmp_list)

# build the dictionary
for i in range(train_size):
    for word in all_lines[i]:
        try:
            vocabulary[word] += 1
        except KeyError:
            vocabulary[word] = 1

# truncate the dictionary
sorted_list = sorted(vocabulary.items(), key=lambda x:x[1])
sorted_list.reverse()
dictionary_size = 8000
truncated_vocabulary: dict = {}
for i in range(dictionary_size - 1):
    word, freq = sorted_list[i]
    truncated_vocabulary[word] = freq
truncated_vocabulary['UNK'] = 0
for i in range(dictionary_size - 1, vocabulary.__len__()):
    _, freq = sorted_list[i]
    truncated_vocabulary['UNK'] += freq

# re-sort the dictionary
sorted_list = sorted(truncated_vocabulary.items(), key=lambda x:x[1])
sorted_list.reverse()
dictionary_size = 8000
truncated_vocabulary: dict = {}
for i in range(dictionary_size - 1):
    word, freq = sorted_list[i]
    truncated_vocabulary[word] = freq

# generate a dictionary map string to IDs
dictionary: dict = {}
idn = 0
for word in truncated_vocabulary:
    dictionary[word] = idn
    idn += 1

# replace less frequent words in all_lines with 'UNK'
tokenized_lines = []
for i in range(train_size):
    tokenized_lines.append([])
    for j in range(len(all_lines[i])):
        if not all_lines[i][j] in truncated_vocabulary:
            tokenized_lines[i].append(dictionary['UNK'])
        else:
            tokenized_lines[i].append(dictionary[all_lines[i][j]])

# build a 4-gram
four_gram: dict = {}
for i in range(len(tokenized_lines)):
    cur_line = tokenized_lines[i]
    cur_len = len(cur_line)
    if (cur_len < 4):
        continue
    for j in range(cur_len-3):
        cur_tuple = (cur_line[j], cur_line[j+1], cur_line[j+2], cur_line[j+3])
        try:
            four_gram[cur_tuple] += 1
        except KeyError:
            four_gram[cur_tuple] = 1

# sort the 4-gram
sorted_list = sorted(four_gram.items(), key=lambda x:x[1])
sorted_list.reverse()

# build the x vector for plotting
x_axis = numpy.arange(sorted_list.__len__())
y_axis = numpy.zeros(sorted_list.__len__())
for i in range(sorted_list.__len__()):
    y_axis[i] = sorted_list[i][1]

plt.figure(1)
line_1, = plt.plot(y_axis, label='4-gram')
plt.xlabel('the ids sorted by the frequency')
plt.ylabel('frequency')
plt.title('4-gram')
plt.savefig(gram_savepath)
plt.close(1)

# invert the key-value pair of dictionary
inv_dictionary = {v: k for k, v in dictionary.items()}

flag_print_most_frequent_grams: bool = True
if flag_print_most_frequent_grams:
    for i in range(50):
        gram_tuple = sorted_list[i][0]
        print(inv_dictionary[gram_tuple[0]],
              inv_dictionary[gram_tuple[1]],
              inv_dictionary[gram_tuple[2]],
              inv_dictionary[gram_tuple[3]],
              sep=' ')

# dump the dictionary for later use
with open(dictionary_dumppath, 'wb+') as f:
    pickle.dump(dictionary, f)
with open(inv_dictionary_dumppath, 'wb+') as f:
    pickle.dump(inv_dictionary, f)



print(truncated_vocabulary['UNK'])






# Phase 2: add START tag and END tag :: TODO

# Phase 3: Compute the number of trainable parameters in the network :: TODO