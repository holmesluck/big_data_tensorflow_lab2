
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
    input1 = tf.transpose(input,[1, 0, 2])
    input2 = tf.reshape(input1,[-1, n_inputs])  
    input3 = tf.split(0, n_steps, input2)
    
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_inputs)   
    #tf.get_variable_scope().reuse_variables()
    print tf.get_variable_scope()
    output, state = tf.nn.rnn(cell, input3, dtype=tf.float32)    
    
    #output = output[-1] 
    return input1, input2, input3

     

out_transpose, out_reshape, out_split= lstm(x_, n_inputs, n_steps)
#outputs_flat = outputs


        
init = tf.initialize_all_variables()
        
with tf.Session() as sess:
    sess.run(init)
    print '----------------------'
    print 'out_transpose =', sess.run(out_transpose, feed_dict={x_:train_input})
    print '----------------------'
    print 'outputs_reshape =', sess.run(out_reshape, feed_dict={x_:train_input})
    print '----------------------'
    print 'out_split =', sess.run(out_split, feed_dict={x_:train_input})


