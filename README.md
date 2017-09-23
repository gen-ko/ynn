# ynn
A repo for 10-707 programming assignments

# Author: Yuan Liu

1. How to test those files?

It is not limited to a specific IDE, using Pycharm or terminal are both OK.

But when launch from terminals, it needs to run from the top level script 'run.py', and import the script that you
would like to run in 'run.py'
using other IDEs are not required to use 'run.py'

3. Minimum Python Version?

3.6


2. How to build a simple NN?

Make sure layer.py and network.py under folder ./src is imported.
Then build a NN like this example:

layers = [layer.Linear(784, 150),
          layer.Sigmoid(150, 150),
          layer.Linear(150, 10),
          layer.Softmax(10, 10)]

myNN = NN(layers, learning_rate=0.1, debug=False, momentum=0.9, regularizer=0.0001)

myNN.train(x_train, y_train, x_valid, y_valid, epoch=300, batch_size=64)
