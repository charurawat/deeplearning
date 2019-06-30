"""
Kaggle Competition: https://www.kaggle.com/c/aerial-cactus-identification/data

This problem is a binary image classification on detecting cactai. 
You are ONLY allowed to use Convolutional Neural Networks in this assignment. 
"""


import pandas as pd
import numpy as np
import tensorflow as tf
import shutil
import pathlib
import random
import os

labels = pd.read_csv('C:/Users/CHARU/Desktop/tflow/assignment/Assignment-3/data/train.csv')

train_data_root = 'C:/Users/CHARU/Desktop/tflow/assignment/Assignment-3/data/train/'
train_data_root = pathlib.Path(train_data_root)
valid_image_paths = []

train_image_paths = list(train_data_root.glob('*'))
train_image_paths = [str(path) for path in train_image_paths]
random.shuffle(train_image_paths)

# Train Valid Split
for img in train_image_paths:
    if random.randint(0, 10) <= 2:
        train_image_paths.remove(img)
        valid_image_paths.append(img)

train_image_count = len(train_image_paths)
valid_image_count = len(valid_image_paths)

# Test data
test_data_root = 'C:/Users/CHARU/Desktop/tflow/assignment/Assignment-3/data/test/'
test_data_root = pathlib.Path(test_data_root)
test_image_paths = list(test_data_root.glob('*'))
test_image_paths = [str(path) for path in test_image_paths]

# generate list of labels
valid_image_labels = [int(labels[labels['id'] == os.path.basename(path)]['has_it']) for path in valid_image_paths]
train_image_labels = [int(labels[labels['id'] == os.path.basename(path)]['has_it']) for path in train_image_paths]


test_image_labels = [0 if random.randint(0, 10) < 5 else 1 for path in test_image_paths]
values = train_image_labels
n_values = np.max(values) + 1

# One hot encoding of  labels
train_image_labels = np.eye(n_values)[values]
values = valid_image_labels 
n_values = np.max(values) + 1

# One hot encoding of image labels
valid_image_labels = np.eye(n_values)[values]
values = test_image_labels
n_values = np.max(values) + 1

# One hot encoding of image labels
test_image_labels = np.eye(n_values)[values]

# normalizing
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [32, 32])
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)
batch_size = 32

 
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))
valid_ds = tf.data.Dataset.from_tensor_slices((valid_image_paths, valid_image_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_image_paths, test_image_labels))

train_image_label_ds = train_ds.map(load_and_preprocess_from_path_label)
train_image_label_ds = train_image_label_ds.batch(batch_size)

valid_image_label_ds = valid_ds.map(load_and_preprocess_from_path_label)
valid_image_label_ds = valid_image_label_ds.batch(batch_size)

test_image_label_ds = test_ds.map(load_and_preprocess_from_path_label)
test_image_label_ds = test_image_label_ds.batch(batch_size)

n_train = len(train_image_paths)
n_val = len(valid_image_paths)
n_test = len(test_image_paths)

# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_image_label_ds.output_types, 
                                           train_image_label_ds.output_shapes)
img, label = iterator.get_next()

# initializer for train_data
train_init = iterator.make_initializer(train_image_label_ds)

# initializer for valid_data
val_init = iterator.make_initializer(valid_image_label_ds)

# initializer for test_data
test_init = iterator.make_initializer(test_image_label_ds)

# Model parameters

# Input
height = 32
width = 32
channels = 3
n_inputs = height * width

# Number of feature maps, size of receptive field, stride, and padding
conv1_fmaps, conv1_ksize, conv1_stride, conv1_pad = 32, 3, 1, "SAME"
conv2_fmaps, conv2_ksize, conv2_stride, conv2_pad = 64, 3, 1, "SAME"
conv3_fmaps, conv3_ksize, conv3_stride, conv3_pad = 128, 3, 1, "SAME"

# Pooling layer, chose 3
max_pool_size = 3
max_pool_stride = 1

# Fully connected layer 
n_fc1 = 128

# Output
n_outputs = 2

# Training
learning_rate = 0.01 # experimented with 0.05,0.001
n_epochs = 10 # experimented with 10,20,30...

# Model
img = tf.reshape(img, shape=[-1, height, width, 3])

# using 3 layers 
conv1 = tf.layers.conv2d(inputs=img, filters=conv1_fmaps, kernel_size=conv1_ksize, 
                         padding=conv1_pad, strides=conv1_stride, activation=tf.nn.relu, name='conv1')

conv2 = tf.layers.conv2d(inputs=conv1, filters=conv2_fmaps, kernel_size=conv2_ksize, 
                         padding=conv2_pad, strides=conv2_stride, activation=tf.nn.relu, name='conv2')

conv3 = tf.layers.conv2d(inputs=conv2, filters=conv3_fmaps, kernel_size=conv3_ksize, 
                         padding=conv3_pad, strides=conv3_stride, activation=tf.nn.relu, name='conv3')

max_pool = tf.layers.max_pooling2d(inputs=conv3, pool_size=max_pool_size, strides=1)

fc1 = tf.contrib.layers.flatten(max_pool)

logits = tf.layers.dense(fc1, 2)

label = tf.reshape(label, shape=[-1, 2])
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
loss = tf.reduce_mean(entropy)
loss_summary = tf.summary.scalar('loss', loss)
# using SGD Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# calculate accuracy with valid set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
accuracy_summary = tf.summary.scalar('accuracy', accuracy)

import time

trn_l = []
val_l = []
trn_a = []
val_a = []

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('C:/Users/CHARU/Desktop/tflow/assignment/Assignment-3/data/graphs', sess.graph)
    
    # train the model n_epochs times
    for i in range(n_epochs):
        sess.run(train_init)
        total_correct_preds = 0
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, accuracy_batch = sess.run([optimizer, loss, accuracy])
                total_correct_preds += accuracy_batch
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
        trn_l.append(total_loss/n_batches)
        trn_a.append(total_correct_preds/n_train)
        
        sess.run(val_init)
        total_correct_preds = 0
        total_loss = 0
        n_batches = 0
        try:
            while True:
                l, accuracy_batch = sess.run([loss, accuracy])
                total_correct_preds += accuracy_batch
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Validation Accuracy {0}'.format(total_correct_preds/n_val))
        val_l.append(total_loss/n_batches)
        val_a.append(total_correct_preds/n_val)
    
    sess.run(test_init)
    test_preds = []
    try:
        while True:
            p = sess.run(tf.argmax(preds, 1))
            test_preds.append(p)
    except tf.errors.OutOfRangeError:
        pass
        
writer.close()

test_predictions = []
for i in range(len(test_preds)):
    test_predictions = test_predictions + list(test_preds[i])
sub = pd.read_csv('input/sample_submission.csv')
sub.head()

sub['id'] = [a.split('/')[4] for a in test_image_paths]
sub['has_cactus'] = test_predictions
sub.to_csv('cactus_predictions.csv', index=None)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics for baseline CNN')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(trn_l)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(trn_a)


import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Validation Metrics for baseline CNN')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(val_l)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(val_a)