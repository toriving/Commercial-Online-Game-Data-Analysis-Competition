import tensorflow as tf
import numpy as np
import pandas as pd
import os, sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]='0'

test1 = np.loadtxt('../preprocess/test1_preprocess_1.csv', delimiter=',')
test2 = np.loadtxt('../preprocess/test2_preprocess_1.csv', delimiter=',')

result1 = test1[:,:3]
result2 = test2[:,:3]
test1 = test1[:,1:]
test2 = test2[:,1:]


tf.reset_default_graph()

tf.set_random_seed(123)
np.random.seed(123)

X = tf.placeholder(tf.float32, [None, 36])
Y_day = tf.placeholder(tf.float32, [None, 1])
Y_money = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32, [])

layer = tf.layers.dense(X, 512, activation='relu')
layer = tf.layers.dense(layer, 256, activation='relu')
layer = tf.layers.dense(layer, 256, activation='relu')
layer = tf.nn.dropout(layer, keep_prob)


day_layer = tf.layers.dense(layer, 128, activation='relu')
day_layer = tf.layers.dense(day_layer, 64, activation='relu')
day_layer = tf.nn.dropout(day_layer, keep_prob)

money_layer = tf.layers.dense(layer, 128, activation='relu')
money_layer = tf.layers.dense(money_layer, 64, activation='relu')
money_layer = tf.nn.dropout(money_layer, keep_prob)

day = tf.layers.dense(day_layer, 1)
day = tf.clip_by_value(day, 0, 64)
money = tf.layers.dense(money_layer, 1)
money = tf.clip_by_value(money, 0, 40)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, '../model/model')
    
    day_predict = sess.run(day, feed_dict={X: test1, keep_prob:1.0})
    money_predict = sess.run(money, feed_dict={X: test1, keep_prob:1.0})
    money_predict[money_predict < 0] = 0

    result1[:, 1] = np.round(day_predict).reshape(-1)
    result1[:, 2] = money_predict.reshape(-1)
    result1[:, 1] = np.where(result1[:, 1] > 64, 64, result1[:, 1])
    result1[:, 1] = np.where(result1[:, 1] < 0, 0, result1[:, 1])

    np.savetxt('test1_predict.csv', result1, delimiter=',', header='acc_id,survival_time,amount_spent', comments='')

    day_predict = sess.run(day, feed_dict={X: test2, keep_prob:1.0})
    money_predict = sess.run(money, feed_dict={X: test2, keep_prob:1.0})
    money_predict[money_predict < 0] = 0

    result2[:, 1] = np.round(day_predict).reshape(-1)
    result2[:, 2] = money_predict.reshape(-1)
    result2[:, 1] = np.where(result2[:, 1] > 64, 64, result2[:, 1])
    result2[:, 1] = np.where(result2[:, 1] < 0, 0, result2[:, 1])

    np.savetxt('test2_predict.csv', result2, delimiter=',', header='acc_id,survival_time,amount_spent', comments='')
    
print('Finish')