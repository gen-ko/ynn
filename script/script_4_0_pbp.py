# required python version: 3.6+

import tensorflow as tf


# naive model

x1 = tf.placeholder(dtype=tf.float32, shape=[None, 28*28], name='x1')

w = tf.Variable(initial_value=tf.random_normal([784, 200], stddev=0.35),
                trainable=True
                    )