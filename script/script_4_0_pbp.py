# required python version: 3.6+

import tensorflow as tf


# naive model

x1 = tf.placeholder(dtype=tf.float32, shape=[None, 28*28], name='x1')

w = tf.Variable(initial_value=tf.random_normal([784, 200], stddev=0.35),
                trainable=True
                    )

b = tf.Variable(initial_value=tf.zeros(shape=200), dtype=tf.float32, name='b1')


y = tf.matmul(x1, w)+b

gw, gb = tf.gradients(ys=y, xs=[w, b])


def pdense(x, units):
    mean = tf.Variable(initial_value=tf.random_normal([784, 200], stddev=0.35),
                    trainable=True
                    )

    var = tf.Variable(initial_value=tf.zeros(shape=[784, 200]), trainable=True)

    w = tf.distributions.Normal(loc=mean, scale=var).sample()



print('h')