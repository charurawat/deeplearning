"""
Author - Charu Rawat
Computing ID: cr4zy
"""

"""
XOR Neural Network Implementation in Tensorflow using the cross-entropy function and the sigmoid function
"""

import tensorflow as tf
import numpy as np
from math import exp, log

#user-defined sigmoid func
def get_sigmoid(x):
  denominator = tf.add(tf.math.exp(x), tf.ones(shape=x.shape.as_list()))
  return tf.divide(tf.math.exp(x), denominator)

#user-defined binary cross entropy
def get_binary_cross_entropy(labels, logits):
  a = tf.multiply(labels, tf.log(logits))
  b = tf.subtract(tf.cast(tf.ones_like(logits), tf.float32), logits)
  d = tf.subtract(tf.cast(tf.ones_like(logits), tf.float32), labels)
  c = tf.multiply(d, tf.log(b))
  return tf.scalar_mul(-1, tf.add(a, c))

X = tf.placeholder(shape=[4, 2], name='X', dtype= tf.int32)
Y = tf.placeholder(shape=[4, 1], name='Y', dtype=tf.int32)

W = tf.Variable(tf.ones([2, 2]), name="W")
w = tf.get_variable(name = "w", initializer=np.array([[1], [-2]]))

c = tf.get_variable(name = "rerror_val", initializer = np.array([[0], [-1]]))
b = tf.cast(tf.Variable(tf.zeros([4, 1]), name="b"), dtype=tf.float32)

# Hidden layer- Sigmoid
h = tf.add(tf.transpose(tf.matmul(tf.cast(X, dtype=tf.float32),
                                          tf.cast(W, dtype=tf.float32))), tf.cast(c, dtype=tf.float32))

zeros = tf.zeros_like(h, dtype=h.dtype)

cond = (h >= zeros)
relu_logits = tf.where(cond, h, zeros)
relu_logits = tf.transpose(tf.matmul(tf.transpose(tf.cast(w, tf.float32)), relu_logits))

h_logit = tf.add(relu_logits, b)

# y_estimated = tf.cast(get_sigmoid(h_logit), dtype=tf.float32)
y_estimated = tf.cast(get_sigmoid(h_logit), dtype=tf.float32)
# Binary cross entropy is the loss function
loss = tf.reduce_mean(get_binary_cross_entropy(tf.cast(Y, dtype=tf.float32), y_estimated))

# optimizer
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

INPUT_XOR = [[0, 0], [0, 1], [1, 0], [1, 1]]
OUTPUT_XOR = [[0], [1], [1], [0]]

with tf.Session() as sess:
  writer = tf.summary.FileWriter('./graphs', sess.graph)
  sess.run(tf.global_variables_initializer())

  # Total number of epochs
  num_epochs = 2000
  for epoch in range(num_epochs):

    sess.run(train_step, feed_dict={X: INPUT_XOR, Y: OUTPUT_XOR})

    if epoch == num_epochs - 1:
      print('Epoch: ', epoch)
      print('y_estimated: ' + str(sess.run(tf.cast(tf.round(y_estimated), dtype=tf.int32), feed_dict={X: INPUT_XOR, Y: OUTPUT_XOR})))
      print('loss: ', str(sess.run(loss, feed_dict={X: INPUT_XOR, Y: OUTPUT_XOR})))
      print('======================' * 2)


  writer.close()