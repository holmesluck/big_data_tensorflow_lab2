#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 07:34:01 2017

### Lab 2 ###

@author: lizhuo

"""
### import libraries ###
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


###############################################################################

### functions ###
def feature_scaling(input_pd, scaling_meathod):
    print  ('feature scaling')
    #scaled_pd = pd.DataFrame()
    if scaling_meathod == 'z-score':
        scaled_pd = (input_pd - input_pd.mean()) / input_pd.std()
    elif scaling_meathod == 'min-max':
        #####################################
        ### filling the missing part here ###
        #####################################
    return scaled_pd
            
def input_reshape(input_pd, start, end, batch_size, batch_shift, n_features):
    print ('*** input reshape ***')
    temp_pd = input_pd[start-1: end+batch_size-1]
    output_pd = map(lambda y : temp_pd[y:y+batch_size], xrange(0, end-start+1, batch_shift))
    output_temp = map(lambda x : np.array(output_pd[x]).reshape([-1]), xrange(len(output_pd)))
    output = np.reshape(output_temp, [-1, batch_size, n_features])
    return output
    
def target_reshape(input_pd, start, end, batch_size, batch_shift, n_step_ahead, m_steps_pred):
    print ('*** target reshape ***')
    temp_pd = input_pd[start+batch_size+n_step_ahead-2: end+batch_size+n_step_ahead+m_steps_pred-2]
    output_pd = map(lambda y : temp_pd[y:y+m_steps_pred], xrange(0, end-start+1, batch_shift))
    output_temp = map(lambda x : np.array(output_pd[x]).reshape([-1]), xrange(len(output_pd)))
    output = np.reshape(output_temp, [-1,1])
    return output

def lstm(input, n_inputs, n_steps, n_of_layers, scope_name): 
    num_layers = n_of_layers
    input = tf.transpose(input,[1, 0, 2])
    input = tf.reshape(input,[-1, n_inputs])  
    input = tf.split(0, n_steps, input)
    with tf.variable_scope(scope_name):
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_inputs) 
        cell = tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers)
        output, state = tf.nn.rnn(cell, input, dtype=tf.float32)    
    output = output[-1] 
    return output    
    
### parameter setting ###
# the selected input features
feature_to_input = ['open price', 'highest price', 'lowest price', 'close price', 'volumn']
# the item we want to predict
feature_to_predict = ['close price']
# the feature needed to do feature scaling
feature_to_scale = ['volumn']
# feature scaling method
sacling_meathod = 'min-max'

# specifify the start and end position of training and testing
train_start = 1 
train_end = 1200
test_start = 1201
test_end = 1400

# the length of a batch (n_steps)
batch_size = 50
# the interval between two batches
batch_shift = 1
# the intervals between the training end and the forecast day
n_step_ahead = 1
m_steps_pred = 1
# how many features that will be input
n_features = len(feature_to_input)

# lstm parameters
# name of the lstm function
lstm_scope_name = 'lstm_prediction'
# number of lstm layers
n_lstm_layers = 1

### regression parameter
n_pred_class = 1

### learning parameter
learning_rate = 0.01

# iterative parameters
# number of iterations
EPOCHS = 1000
# the interval of printing validation result
PRINT_STEP = 100

### read data and data preprocessings ###
# read .csv file by pd.read_csv command, and the results will be stored as pandas DataFrame 
read_data_pd = pd.read_csv('./stock_price.csv')
# drop the column 'year', 'month', 'day', in other words, keep the column 'open prince', 'highest price', 'lowest price' and 'close price'
input_pd = read_data_pd.drop(['year','month','day'], axis=1)
# call the function feature scaling
temp_pd = feature_scaling(input_pd[feature_to_scale],sacling_meathod)
input_pd[feature_to_scale] = temp_pd



### construct the training data ###
train_input_temp_pd = input_pd[feature_to_input]
# call the 'input_reshape' function to construct the training input data (a matrix)
train_input_nparr = input_reshape(train_input_temp_pd, train_start, train_end, batch_size, batch_shift, n_features)

train_target_temp_pd = input_pd[feature_to_predict]
# call the 'target_reshape' function to construct the training target data (label) (a column vector)
train_target_nparr = target_reshape(train_target_temp_pd, train_start, train_end, batch_size, batch_shift, n_step_ahead, m_steps_pred)

### construct the testing data ###
### the name of test input and test target is 'train_input_nparr' and 'train_target_nparr' respectively
    #####################################
    ### filling the missing part here ###
    #####################################
test_input_nparr = ...

test_target_nparr = ...


######### main body #########
# reset the graph
tf.reset_default_graph()

# difine two place holders, which can be reagarded as handles pointing to a memory space, 
# but actually the memory has not been allocated to it before start the session.

# x_ has the size of 3-dimension, None means the 1st dimension can be of any length, 
# the 2nd and 3rd dimension are of size batch_size and n_features
x_ = tf.placeholder(tf.float32, [None, batch_size, n_features])
# the same as x_ but it is a placeholder of 2-dimension
y_ = tf.placeholder(tf.float32, [None, 1])

### call the lstm-rnn function ###
lstm_output = lstm(x_, n_features, batch_size, n_lstm_layers, lstm_scope_name)

### linear regressor ###
### define the linear regressor Y = WX + b
    #####################################
    ### filling the missing part here ###
    #####################################
W = ...
b = ...
y = ...

### define the cost function (loss function)
### define the cost function with mean square error
    #####################################
    ### filling the missing part here ###
    #####################################
cost_func = ...

### define the learning algorithm (optimizer), such as gradient descent, 
# stochastic gradient, RMSP, Adam.
    #####################################
    ### filling the missing part here ###
    #####################################
train_op = ...

### start the session() ###

# initialize all variables
init = tf.initialize_all_variables()        
with tf.Session() as sess:
    sess.run(init)

    for ii in range(EPOCHS):
        # train the model
        sess.run(train_op, feed_dict={x_:train_input_nparr, y_:train_target_nparr})
        if ii % PRINT_STEP == 0:
            # print the training error periorically
            cost = sess.run(cost_func, feed_dict={x_:train_input_nparr, y_:train_target_nparr})
            print ('iteration =', ii, 'training cost:', cost)

    print ('-------------- validation ----------------')
    validation = sess.run(y, feed_dict={x_:train_input_nparr})
    ### calculate mse, rmse, mae, mape of validation results
    #####################################
    ### filling the missing part here ###
    #####################################
    
    validation_mse = '...'
    validation_rmse = '...'
    validation_mae = '...'
    validation_mape = '...'
    print ('validation_mse =', validation_mse)
    print ('validation_rmse =', validation_rmse)
    print ('validation_mae =', validation_mae)
    print ('validation_mape =', validation_mape)

    print ('-------------- prediction ----------------')
    prediction = sess.run(y, feed_dict={x_:test_input_nparr})
    ### calculate mse, rmse, mae, mape of prediction results
    #####################################
    ### filling the missing part here ###
    #####################################
    prediction_mse = '...'
    prediction_rmse = '...'
    prediction_mae = '...'
    prediction_mape = '...'
    print ('prediction_mse =', prediction_mse)
    print ('prediction_rmse =', prediction_rmse)
    print ('prediction_mae =', prediction_mae)
    print ('prediction_mape =', prediction_mape)
    
    
plt.figure(1)
plt.plot(train_target_nparr,color='blue',linewidth=1.5)
plt.hold
plt.plot(validation, color='green',linewidth=1.5)
plt.title('Compare the Real Stock Price and the Validation Results',fontsize=18)
plt.xlabel('Day Index',fontsize=17) 
plt.ylabel('Price',fontsize=17)   
plt.legend(['real price', 'validation'],fontsize=17)
plt.show()

    
plt.figure(2)
### plot the figure of compare real stock price and the prediction results 
# the same as above, but use different line colors 
#####################################
### filling the missing part here ###
#####################################
