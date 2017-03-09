# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 20:09:06 2016
dynamic_rnn
@author: lizhuo
"""
### verification ### ### ###

import numpy as np
import tensorflow as tf

EPOCHS = 10000
PRINT_STEP = 1000

data = np.array([[[1, 2, 3, 4, 5], 
                  [2, 3, 4, 5, 6], 
                  [3, 4, 5, 6, 7], 
                  [4, 5, 6, 7, 8]], 
                 [[5, 6, 7, 8, 9], 
                  [6, 7, 8, 9, 10], 
                  [7, 8, 9, 10, 11],
                  [8, 9, 10, 11, 12]],
                 [[9, 10, 11, 12, 13], 
                  [10, 11, 12, 13, 14], 
                  [11, 12, 13, 14, 15],
                  [12, 13, 14, 15, 16]]])
target = np.array([[[6], 
                    [7], 
                    [8], 
                    [9]], 
                   [[10], 
                    [11], 
                    [12],
                    [13]],
                   [[14], 
                    [15], 
                    [16],
                    [17]]])


n_inputs = data.shape[2]
n_steps = data.shape[1]
n_classes = 1
#tf.get_variable_scope().reuse_variables()
x_ = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y_ = tf.placeholder(tf.float32, [None, 1])
tf.get_variable_scope().reuse_variables()
    


def lstm(input, n_inputs, n_steps):  
    input = tf.transpose(input,[1, 0, 2])
    input = tf.reshape(input,[-1, n_inputs])  
    input = tf.split(0, n_steps, input)
    
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_inputs)   
    #tf.get_variable_scope().reuse_variables()
    print tf.get_variable_scope()
    output, state = tf.nn.rnn(cell, input, dtype=tf.float32)    
    #output = output[-1] 
    return output

#input_shape = data.shape[-1]

outputs = lstm(x_, n_inputs, n_steps)
outputs_flat = tf.reshape(outputs, [-1, n_inputs])

W = tf.Variable(tf.random_normal([n_inputs, n_classes]))     
b = tf.Variable(tf.random_normal([n_classes]))
y = tf.matmul(outputs_flat, W) + b

target_flat = target.reshape([-1, 1])

cost = tf.reduce_mean(tf.square(y - y_))
#train_op = tf.train.RMSPropOptimizer(0.001, 0.2).minimize(cost)
train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
#train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
        
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    #print sess.run(target_flat)
    for i in range(EPOCHS):
        #print sess.run(output, feed_dict={x_:data})
        sess.run(train_op, feed_dict={x_:data, y_:target_flat})
        if i % PRINT_STEP == 0:
            c = sess.run(cost, feed_dict={x_:data, y_:target_flat})
            print('training cost:', c)

    response = sess.run(y, feed_dict={x_:data})
    print(response)
