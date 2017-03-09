
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 01:05:58 2016

@author: lizhuo
"""

###
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 03:47:32 2016

@author: lizhuo
"""

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

train_input = np.array([[[1, 2, 3, 4, 5], 
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

train_target = np.array([[[6], 
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

test_input = np.array([[[13, 14, 15, 16, 17], 
                        [14, 15, 16, 17, 18], 
                        [15, 16, 17, 18, 19],
                        [16, 17, 18, 19, 20]]])

tf.reset_default_graph()

n_inputs = train_input.shape[2]
n_steps = train_input.shape[1]
n_classes = 1

x_ = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y_ = tf.placeholder(tf.float32, [None, 1])

    

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

     

outputs = lstm(x_, n_inputs, n_steps)
#outputs_flat = outputs
outputs_flat = tf.reshape(outputs, [-1, n_inputs])

W = tf.Variable(tf.random_normal([n_inputs, n_classes]))                                                                                                    
b = tf.Variable(tf.random_normal([n_classes]))
y = tf.matmul(outputs_flat, W) + b
                                                                                                                                                
train_target = train_target.reshape([-1, 1])


cost_func = tf.reduce_mean(tf.square(y - y_))
#train_op = tf.train.RMSPropOptimizer(0.001, 0.2).minimize(cost)
#train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
train_op = tf.train.AdamOptimizer(0.01).minimize(cost_func)


        
init = tf.initialize_all_variables()
        
with tf.Session() as sess:
    sess.run(init)
    #print '---------------------- outputs =', sess.run(outputs, feed_dict={x_:train_input})
    #print '---------------------- outputs_flat =', sess.run(outputs_flat, feed_dict={x_:train_input})
    #print '---------------------- outputs_flat =', sess.run(y, feed_dict={x_:train_input})

    for ii in range(EPOCHS):
        #print sess.run(output, feed_dict={x_:data})
        sess.run(train_op, feed_dict={x_:train_input, y_:train_target})
        if ii % PRINT_STEP == 0:
            cost = sess.run(cost_func, feed_dict={x_:train_input, y_:train_target})
            print 'iteration =', ii, 'training cost:', cost

    print '-------------- validation ----------------'
    validation = sess.run(y, feed_dict={x_:train_input})
    print 'validation =', '\n'
    print validation
    print '-------------- prediction ----------------'
    prediction = sess.run(y, feed_dict={x_:test_input})
    print 'prediction =', '\n'
    print prediction
