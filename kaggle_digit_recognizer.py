"""
Code for Kaggle Competition  - Digit Recognizer
Highest acc achieved - 98.9% (submission6 - charurawat)
"""

#%%
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import time

#Reading train
dir = os.path.join(os.getcwd(), 'digit-recognizer')
train_df = pd.read_csv(os.path.join(dir, 'train.csv'))
print(train_df.shape)

#Reading test
test_df = pd.read_csv(os.path.join(dir, 'test.csv'))
print(test_df.shape)

#split data into train vs valid
train_split = train_df.sample(frac=0.8,random_state=200)
valid_split = train_df.drop(train_split.index)

# one-hot encoding
def one_hot_label(x):
    result = [0 for i in range(10)]
    result[x] = 1
    return result

train_split['label'] = train_split['label'].apply(one_hot_label)
valid_split['label'] = valid_split['label'].apply(one_hot_label)

train_labels_df = pd.DataFrame(dict(zip(train_split['label'].index, train_split['label'].values))).transpose()
valid_labels_df = pd.DataFrame(dict(zip(valid_split['label'].index, valid_split['label'].values))).transpose()

# Define paramaters for the model
learning_rate = 0.001 # change to experiment
batch_size = 128
n_epochs = 30 # change to experiment
n_train = train_labels_df.shape[0]
n_test = valid_labels_df.shape[0]


train = (train_split.iloc[:, 1: 785].values.astype(float), train_labels_df[:].values.astype(float))
valid = (valid_split.iloc[:, 1: 785].values.astype(float), valid_labels_df[:].values.astype(float))

train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.batch(batch_size)

# create test and batch it
test_data = tf.data.Dataset.from_tensor_slices(valid)
test_data = test_data.batch(batch_size)

# create one iterator and initialize it
iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)  # initializer for train
test_init = iterator.make_initializer(test_data)  # initializer for test

# create weights and bias
### Creating a hidden layer with a size of 400
w2 = tf.get_variable(initializer=tf.random_normal(shape = [img.shape[1].value, 400], mean = 0, stddev= 0.01),name = 'weight_2')
b2 = tf.get_variable(initializer=tf.zeros(shape = [400]), name = 'bias_2')
hidden_layer2 = tf.add(tf.matmul(tf.cast(img, tf.float32), tf.cast(w2, tf.float32)), tf.cast(b2, tf.float32))
#Rectified Linear Unit is chosen as the activation function
hidden_layer_activation_2 = tf.nn.relu(hidden_layer2)

### Creating a hidden layer with a size of 100
w4 = tf.get_variable(initializer=tf.random_normal(shape = [400, 100], mean = 0, stddev= 0.01),name = 'weight_4')
b4 = tf.get_variable(initializer=tf.zeros(shape = [100]), name = 'bias_4')
hidden_layer4 = tf.add(tf.matmul(tf.cast(hidden_layer_activation_2, tf.float32), tf.cast(w4, tf.float32)), tf.cast(b4, tf.float32))
#Rectified Linear Unit is chosen as the activation function
hidden_layer_activation_4 = tf.nn.relu(hidden_layer4)

######## Output Layer
w5 = tf.get_variable(initializer=tf.random_normal(shape = [100,label.shape[1].value], mean = 0, stddev= 0.01),name = 'weight_5')
b5 = tf.get_variable(initializer=tf.zeros(shape = [1, label.shape[1].value]), name = 'bias_5')
logits = tf.add(tf.matmul(hidden_layer_activation_4, w5), b5)

# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
loss = tf.reduce_mean(entropy)

# using Gradient Descent Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
sess =  tf.Session()
start_time = time.time()
sess.run(tf.global_variables_initializer())

# train the model n_epochs times
for i in range(n_epochs):
  sess.run(train_init) 
  total_loss = 0
  n_batches = 0
  try:
    while True:
      _, batch_loss = sess.run([optimizer, loss])
      total_loss += batch_loss
      n_batches += 1
  except tf.errors.OutOfRangeError:
    pass
  print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))

print('Total time: {0} seconds'.format(time.time() - start_time))

# test the model
sess.run(test_init) 
total_correct_preds = 0
try:
  while True:
    accuracy_batch = sess.run(accuracy)
    total_correct_preds += accuracy_batch
except tf.errors.OutOfRangeError:
  pass

print('Accuracy {0}'.format(total_correct_preds / n_test))
#Accuracy = 98.9
# writer.close()

#%% Prediction time
img.predict = test_df.iloc[:, 0: 784].values.astype(float)
img.predict = tf.Variable(img.predict, name='Predict_data')

prediction_h1_layer = tf.add(tf.matmul(tf.cast(img.predict, tf.float32), w2), b2)
#Rectified Linear Unit is chosen as the activation function
prediction_h1_layer = tf.nn.relu(prediction_h1_layer)

prediction_h2_layer = tf.add(tf.matmul(tf.cast(prediction_h1_layer, tf.float32), tf.cast(w4, tf.float32)), tf.cast(b4, tf.float32))
#Rectified Linear Unit is chosen as the activation function
prediction_h2_layer = tf.nn.relu(prediction_h2_layer)

######## Output Layer
logits = tf.add(tf.matmul(prediction_h2_layer, w5), b5)
preds = tf.nn.softmax(logits)
imageId = [i for i in range(1, test_df.shape[0] + 1)]
final_preds = tf.argmax(preds, 1)

sess.run(img.predict.initializer)
#%% Get the predictions as numbers
prediction_label = sess.run(final_preds)

#%% Generating submission csv
finalDf = pd.DataFrame({'ImageId': imageId, 'Label':prediction_label})

finalDf.to_csv('submission6.csv', index=False)


#%% Close session
sess.close()
