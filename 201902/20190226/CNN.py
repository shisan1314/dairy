
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1]) #batch, length, width, channel

W_conv1 = weight_variable([5,5,1,32]) #filter_length, filter_width, in_channel, out_channel
b_conv1 = bias_variable([32])#out_channel
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全连接层
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = weight_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(50):
  batch_xs, batch_ys = mnist.test.next_batch(300)
  _, loss = sess.run([train_step,cross_entropy], feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:0.2})
  if i %10 == 0:
    print("loss", loss)


saver = tf.train.Saver()
save_path = saver.save(sess, "save_net.ckpt")
new_saver = tf.train.Saver()
with tf.Session() as sess:
  new_saver.restore(sess, "save_net.ckpt")
  print(sess.run("Variable_6:0"))






