from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):
  Weights = tf.Variable(tf.random_normal([in_size, out_size]))
  biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
  Wx_plus_b = tf.matmul(inputs, Weights) + biases
  if activation_function is None:
      outputs = Wx_plus_b
  else:
      outputs = activation_function(Wx_plus_b)
  return outputs

prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
reduction_indices=[1])) # loss

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_xs, batch_ys = mnist.train.next_batch(100)
_, loss = sess.run([train_step, cross_entropy], feed_dict={xs: batch_xs, ys: batch_ys})
print(loss)