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
data_train_dump_filename = 'train.dump'
data_valid_dump_filename = 'valid.dump'

data_train_filepath = os.path.join(path, data_filepath, data_train_filename)
data_valid_filepath = os.path.join(path, data_filepath, data_valid_filename)
gram_savepath = os.path.join(path, save_filepath, gram_name)
dictionary_dumppath = os.path.join(path, dump_filepath, dictionary_name)
inv_dictionary_dumppath = os.path.join(path, dump_filepath, inv_dictionary_name)
data_train_dump_filepath = os.path.join(path, dump_filepath, data_train_dump_filename)
data_valid_dump_filepath = os.path.join(path, dump_filepath, data_valid_dump_filename)

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
def split_lins(data_input):
    all_lines: list = []
    train_size = data_input.size
    # will serve as a lookup table for the words and their corresponding id.
    for i in range(train_size):
        tmp_list = data_input[i].lower().split()
        tmp_list.insert(0, 'START')
        tmp_list.append('END')
        all_lines.insert(i,  tmp_list)
    return all_lines

all_lines_train = split_lins(data_train)
all_lines_valid = split_lins(data_valid)


# build the dictionary
def build_vocabulary(all_lines: list):
    vocabulary: dict = {}
    train_size = len(all_lines)
    for i in range(train_size):
        for word in all_lines[i]:
            try:
                vocabulary[word] += 1
            except KeyError:
                vocabulary[word] = 1
    return vocabulary


vocabulary = build_vocabulary(all_lines_train)

# truncate the dictionary
def truncate_dictionary(dictionary: dict, size: int):
    sorted_list = sorted(vocabulary.items(), key=lambda x:x[1])
    sorted_list.reverse()
    dictionary_size = size
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
    return truncated_vocabulary

truncated_vocabulary = truncate_dictionary(vocabulary, 8000)

# generate a dictionary map string to IDs
def gen_word_to_id_dict(vocabulary):
    dictionary: dict = {}
    idn = 0
    for word in truncated_vocabulary:
        dictionary[word] = idn
        idn += 1
    return dictionary

dict_word_to_id = gen_word_to_id_dict(truncated_vocabulary)


# replace less frequent words in all_lines with 'UNK'
def replace_with_unk(all_lines, dict_word_to_id):
    tokenized_lines = []
    train_size = len(all_lines)
    for i in range(train_size):
        tokenized_lines.append([])
        for j in range(len(all_lines[i])):
            if not all_lines[i][j] in truncated_vocabulary:
                tokenized_lines[i].append(dict_word_to_id['UNK'])
            else:
                tokenized_lines[i].append(dict_word_to_id[all_lines[i][j]])
    return tokenized_lines

tokenized_lines_train = replace_with_unk(all_lines_train, dict_word_to_id)
tokenized_lines_valid = replace_with_unk(all_lines_valid, dict_word_to_id)


# build a 4-gram
def build_four_gram(tokenized_lines):
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
    return sorted_list

four_gram_train = build_four_gram(tokenized_lines_train)
four_gram_valid = build_four_gram(tokenized_lines_valid)


# plot the 4-gram
def plot_four_gram(four_gram: list):
    x_axis = numpy.arange(four_gram.__len__())
    y_axis = numpy.zeros(four_gram.__len__())
    for i in range(four_gram.__len__()):
        y_axis[i] = four_gram[i][1]

    plt.figure(1)
    line_1, = plt.plot(y_axis, label='4-gram')
    plt.xlabel('the ids sorted by the frequency')
    plt.ylabel('frequency')
    plt.title('4-gram')
    plt.savefig(gram_savepath)
    plt.close(1)
    return

flag_plot_four_gram = True
if flag_plot_four_gram:
    plot_four_gram(four_gram_train)

# invert the key-value pair of dictionary
inv_dictionary = {v: k for k, v in dict_word_to_id.items()}


def print_top_four_gram(four_gram: list, top_num: int):
    for i in range(top_num):
        gram_tuple = four_gram[i][0]
        print(inv_dictionary[gram_tuple[0]],
              inv_dictionary[gram_tuple[1]],
              inv_dictionary[gram_tuple[2]],
              inv_dictionary[gram_tuple[3]],
              sep=' ')
    return

flag_print_most_frequent_grams: bool = False
if flag_print_most_frequent_grams:
    print_top_four_gram(four_gram_train, 50)

# dump the dictionary for later use
with open(dictionary_dumppath, 'wb+') as f:
    pickle.dump(dict_word_to_id, f)
with open(inv_dictionary_dumppath, 'wb+') as f:
    pickle.dump(inv_dictionary, f)

# generate the one-hot representation of inputs
# get the number of the inputs
def process_four_gram(four_gram):
    num_input: int = len(four_gram)
    X: numpy.ndarray
    for i in range(num_input):
        tup = four_gram[i]
        w1 = tup[0][0]
        w2 = tup[0][1]
        w3 = tup[0][2]
        w4 = tup[0][3]

        for j in range(tup[1]):
            array = numpy.array([w1, w2, w3, w4]).reshape(1, 4)
            try:
                X = numpy.concatenate((X, array), axis=0)
            except NameError:
                X = array
    return X

X_train = process_four_gram(four_gram_train)
X_valid = process_four_gram(four_gram_valid)

# dump the one-hot representation of input
with open(data_train_dump_filepath, 'wb+') as f:
    pickle.dump(X_train, f)
with open(data_valid_dump_filepath, 'wb+') as f:
    pickle.dump(X_valid, f)

print(truncated_vocabulary['UNK'])



# Phase 3: Compute the number of trainable parameters in the network
flag_print_num_trainable: bool = True
if flag_print_num_trainable:
    print(8000 * 16 + 128 * 16 * 3 + 128 + 8000 * 128 + 8000)