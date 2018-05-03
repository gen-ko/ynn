# ynn
A lightweight neural network framework

Under development...

Minimum Python Interpreter Requirement:
~~~~
python 3.6+
~~~~

How to build a simple Multilayer Perceptron?
~~~~        
layers = [layer.Linear(784, 150),
          layer.Sigmoid(150, 150),
          layer.Linear(150, 10),
          layer.Softmax(10, 10)]
~~~~
How to construct a neural network?
~~~~
myNN = NN(layers, learning_rate=0.1, 
                  debug=False,
                  momentum=0.9, 
                  regularizer=0.0001)
~~~~
How to train it?
~~~~
myNN.train(x_train,
           y_train, 
           x_valid, 
           y_valid, 
           epoch=300, 
           batch_size=64)
~~~~





