"""
Author - Charu Rawat
Computing ID: cr4zy
"""

"""
XOR Neural Network Implementation in Tensorflow 
"""

# %%
# Import necessary packages
import tensorflow as tf
import numpy as np

# %%
# define shape
X = tf.placeholder(shape=[4, 2], name='X', dtype= tf.int32)
Y = tf.placeholder(shape=[4, 1], name='Y', dtype=tf.int32)

W = tf.Variable(tf.ones([2, 2]), name="W")
w = tf.get_variable(name = "w", initializer=np.array([[1], [-2]]))

c = tf.get_variable(name = "c_val", initializer = np.array([[0], [-1], [0], [-1]]))
b = tf.Variable(tf.zeros([4, 1]), name="b")

# Hidden layer
# using relu activation function
h = tf.nn.relu(tf.add(tf.matmul(tf.matmul(tf.cast(X, dtype=tf.float32),
                                          tf.cast(W, dtype=tf.float32)), tf.cast(w, dtype=tf.float32)),
                      tf.cast(c, dtype=tf.float32)))

# y_prediction
y_estimated = tf.add(h, b)

# Returns mean of (x - y)(x - y) element-wise
loss = tf.reduce_mean(tf.squared_difference(y_estimated, tf.cast(Y, dtype=tf.float32)))

# optimizer
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

INPUT_XOR = [[0, 0], [0, 1], [1, 0], [1, 1]]
OUTPUT_XOR = [[0], [1], [1], [0]]

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  # Total number of epochs
  num_epochs = 200
  for epoch in range(num_epochs):

    sess.run(train_step, feed_dict={X: INPUT_XOR, Y: OUTPUT_XOR})

    print('Epoch: ', epoch)
    print('y_estimated: ' + str(sess.run(tf.cast(tf.round(y_estimated), dtype=tf.int32), feed_dict={X: INPUT_XOR, Y: OUTPUT_XOR})))
    print('loss: ', str(sess.run(loss, feed_dict={X: INPUT_XOR, Y: OUTPUT_XOR})))
    print('======================' * 2)

