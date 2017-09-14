import os
import sys
from src import load_data
from src import plot_data
import matplotlib.pyplot as plt


#  format of data
# disitstrain.txt contains 3000 lines, each line 785 numbers, comma delimited

data_filepath = '../data'
data_train_filename = 'digitstrain.txt'
data_valid_filename = 'digitsvalid.txt'
data_test_filename = 'digitstest.txt'

data_train_filepath = os.path.join(data_filepath, data_train_filename)
data_valid_filepath = os.path.join(data_filepath, data_valid_filename)
data_test_filepath = os.path.join(data_filepath, data_test_filename)


# x range [0, 1]
x_train, y_train = load_data.load_from_path(data_train_filepath)

print(x_train.shape)
print(y_train.shape)
print(y_train)

x_train_reshaped = plot_data.reshape_row_major(x_train[2550], 28, 28)
# plt.imshow(x_train_reshaped)

# so data is row majored
plot_data.plot_image(x_train_reshaped)
plt.show()