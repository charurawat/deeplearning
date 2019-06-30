"""
Author - Charu Rawat
Computing ID: cr4zy

Multiclass classification on detecting cuisine based on list of ingredients (using word embeddings as well)

"""
# load necessary
import os
import pandas as pd
from collections import defaultdict
from collections import Counter
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.model_selection import train_test_split

# Load train and test
train = pd.read_json('./train.json/train.json')
test = pd.read_json('./test.json/test.json')

#Train data
train_data =  train.sample(frac=0.7,random_state=0)

#valid_data
valid_index = [each for each in train.index if each not in train_data.index]
valid_data = train.iloc[valid_index,:]

ingr = [each.lower() for each_ing in train_data.ingredients for each in each_ing]
counter = 0
ingreds_id = defaultdict(int)
id_ingreds = defaultdict(int)
minimum_freq = 2
ing_count = Counter(ingr)

for each in set(ingr):
    if(ing_count[each] < minimum_freq):
        continue
    ingreds_id[each] = counter
    id_ingreds[counter] = each
    counter += 1

ingreds_id['<UNK>'] = len(ingreds_id)
id_ingreds[len(id_ingreds)] = '<UNK>'


# Encode all
train_data['ingredients_code'] = train_data.ingredients.apply(lambda x: [ingreds_id[each.lower()] if each.lower() in ingreds_id else ingreds_id['<UNK>'] for each in x])
valid_data['ingredients_code'] = valid_data.ingredients.apply(lambda x: [ingreds_id[each.lower()] if each.lower() in ingreds_id else ingreds_id['<UNK>'] for each in x])
test['ingredients_code'] = test.ingredients.apply(lambda x: [ingreds_id[each.lower()] if each.lower() in ingreds_id else ingreds_id['<UNK>'] for each in x])
train_data['cuisine_codes'] = train_data['cuisine'].astype('category').cat.codes
valid_data['cuisine_codes'] = valid_data['cuisine'].astype('category').cat.codes

max_ingredients = 10
pad_code = [len(ingreds_id)]

# Apply padding if the ingredient len < max limit
train_data['ingredients_code_pad'] = train_data.ingredients_code.apply(
                                lambda x: pad_code*(max_ingredients - len(x)) + x 
                                    if len(x) < max_ingredients else x[:max_ingredients] )


valid_data['ingredients_code_pad'] = valid_data.ingredients_code.apply(
                                lambda x: pad_code*(max_ingredients - len(x)) + x 
                                    if len(x) < max_ingredients else x[:max_ingredients] )

test['ingredients_code_pad'] = test.ingredients_code.apply(
                                lambda x:  pad_code*(max_ingredients - len(x)) + x
                                    if len(x) < max_ingredients else x[:max_ingredients] )


# model architecture
vocabulary_size = len(ingreds_id)+1
embedding_size = 100
n_neurons = 50 # change and see
n_outputs = 20
batch_size = 64
learning_rate = 0.05 # play around

X = tf.placeholder(tf.int32, [batch_size, max_ingredients])
y = tf.placeholder(tf.int32, [batch_size])


# WORD EMBEDDING
word_emb = tf.get_variable('word_embedding_values', [vocabulary_size, embedding_size], )
embedded_word_ids = tf.nn.embedding_lookup(word_emb,  X)
# RNN 
basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons,name = 'Basic_RNN_block')
outputs, states = tf.nn.dynamic_rnn(basic_cell, embedded_word_ids, dtype=tf.float32)
# output 
logits_out = tf.layers.dense(states, n_outputs)
# applying sparse cross entropy
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                          logits=logits_out)

loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits_out, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

def shuffle_batch(train,X_col, y_col, batch_size,offset,data = 'train'):
    
    X  = train[X_col].iloc[(offset * batch_size) : ((offset+1) * batch_size) ].tolist()
    if(data == 'train'):
        y = train[y_col].iloc[(offset * batch_size) : ((offset+1) * batch_size) ].values
    else:
        y = []
    return(np.array(X),y)

n_epochs = 100

acc_train_all = []
loss_train_all = []
acc_valid_all = []
loss_valid_all = []
acc_train_temp = []
acc_valid_temp = []
loss_train_temp = []
loss_valid_temp = []


with tf.Session() as sess:
    init.run()  
    for epoch in range(n_epochs):
        
        
        offset = 0
        offset_valid = 0
        
        
        #train
        while(offset * batch_size < train_data.shape[0]):
            X_batch, y_batch = shuffle_batch(train_data, 'ingredients_code_pad','cuisine_codes',batch_size,offset)
            
            if(X_batch.shape[0] < batch_size):
                break
                
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            offset += 1

            acc_train_temp.append(accuracy.eval(feed_dict={X: X_batch, y: y_batch}))
            loss_train_temp.append(loss.eval(feed_dict={X: X_batch, y: y_batch}))
    
    
        # validation
        while(offset_valid * batch_size < valid_data.shape[0]):

            #print(offset)
            X_batch_valid, y_batch_valid = shuffle_batch(valid_data, 'ingredients_code_pad', 'cuisine_codes',batch_size,offset_valid)

            if(X_batch_valid.shape[0] < batch_size):
                break

            offset_valid += 1

            acc_valid_temp.append(accuracy.eval(feed_dict={X: X_batch_valid, y: y_batch_valid}))
            loss_valid_temp.append(loss.eval(feed_dict={X: X_batch_valid, y: y_batch_valid}))

            
            
        acc_train_all.append(np.mean(acc_train_temp))
        acc_valid_all.append(np.mean(acc_valid_temp))
        
        loss_train_all.append(np.mean(loss_train_temp))
        loss_valid_all.append(np.mean(loss_valid_temp))

        print("Epoch:", epoch, "Train Accuracy:", np.mean(acc_train_temp),
                  "Train Loss:",np.mean(loss_train_temp))

        print("Epoch:", epoch, "Validation Accuracy:", np.mean(acc_valid_temp),
              "Validation Loss:",np.mean(loss_valid_temp))
        print("\n")
        
            
    # test preds
    offset_test = 0
    test_preds = []
    last_batch = False
    
    while(offset_test * batch_size < test.shape[0]):
            #print(offset)
            X_batch, _ = shuffle_batch(test, 'ingredients_code_pad','cuisine_codes',batch_size,offset_test,'test')
            
            if(X_batch.shape[0] < batch_size):
                pad_seq = [pad_code*max_ingredients] * (batch_size - X_batch.shape[0])
                pad_seq_len = len(pad_seq)
                
                pad_seq.extend(X_batch.tolist())
                X_batch = pad_seq
                last_batch = True
                
            pred = sess.run(logits_out, feed_dict={X: X_batch})
            
            if last_batch:
                    pred = pred[pad_seq_len:]
            test_preds.extend(pred)
            offset_test += 1


# plotting
import matplotlib.pyplot as plt
plt.plot(loss_train_all)
plt.plot(loss_valid_all)
plt.ylabel("Loss")
plt.xlabel("Epochs")

plt.plot(acc_train_all)
plt.plot(acc_valid_all)
plt.ylabel("Loss")
plt.xlabel("Accuracy")

categories = train['cuisine'].astype('category').cat.categories
test_pred_vals = np.argmax(test_preds,axis = 1)
final_predictions = [categories[each] for each in test_pred_vals]
test['cuisine'] = final_predictions
test[['id','cuisine']].to_csv('Kaggle_submission.csv',index = False)
