# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:10:12 2019

@author: Danish
"""

import tensorflow as tf
from JSCC_Methods import  Show_progress
from BatchMaker import BatchMaker
import time
import os
import json
import math
import numpy as np
import sys

def conv2d_layer(input, input_channels, filters, kernel_size, stride, padding, name):
    """ Parameters
        ----------
        **input**: Input Images
    
        **input_channels**: Number of channels in the image e.g 1 or 3. If the layer is not the 1st one,
        then the number of filters in previous layer
        
        **filters**: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution). 
        
        **kernel_size**:  An integer, specifying the height and width of the 2D convolution window, specify 
        the same value for all spatial dimensions.
        
        **stride**:  An integer specifying the strides of the convolution along the height and width.
        
        **padding**: Either the string "SAME" or "VALID  indicating the type of padding algorithm to use.
        
        **name**: Name for the Convolutional Layer.
        
        Returns
        ----------
        **A `Tensor`. Has the same `type` as input.**
        """
    with tf.name_scope(name):
        # Shape of the filter-weights for the convolution
        """For the tf.nn.conv_2d the filter variable should be - A 4-D tensor of shape
           [filter_height, filter_width, in_channels, out_channels/filters] """
        shape = [kernel_size, kernel_size, input_channels, filters]
        #print(shape)
        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.random.truncated_normal(shape, stddev=0.05), name='W')

        # Create new biases, one for each filter
        biases = tf.Variable(tf.constant(0.05, shape=[filters]), name='b')

        # TensorFlow operation for convolution
        layer = tf.nn.conv2d(input=input, filter=weights, strides=stride, padding=padding, name='nn.conv2d')

        # Add the biases to the results of the convolution.
        layer += biases
        
        #tf.compat.v1.summary.histogram('weights', weights)
        #tf.compat.v1.summary.histogram('biases', biases)
        return layer
 
           
def conv2dTranspose_layer(input, input_channels, filters, kernel_size, output_shape, stride, padding, name, activation=None):
    """ Parameters
        ----------
        **input**: Input Images
    
        **input_channels**: Number of channels in the image e.g 1 or 3. If the layer is not the 1st one,
        then the number of filters in previous layer
        
        **filters**: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution). 
        
        **kernel_size**:  An integer, specifying the height and width of the 2D convolution window, specify 
        the same value for all spatial dimensions.
        
        **output_shape**: A 1-D Tensor representing the output shape of the deconvolution op.
        
        **stride**:  An integer specifying the strides of the convolution along the height and width.
        
        **padding**: Either the string "SAME" or "VALID  indicating the type of padding algorithm to use.
        
        **name**: Name for the Transposed Convolutional Layer.
        
        **activation**: Activation function, by default is None, and right now only 'sigmoid' is available. 
        Pass the name 'sigmoid' as a string.
        
        Returns
        ----------
        **A `Tensor`. Has the same `type` as input.**
        """    
    with tf.name_scope(name):
        # Shape of the filter-weights for the convolution
        '''For the tf.nn.conv_2d_transpose the filter variable should be - A 4-D tensor of shape 
          [filter_height, filter_width, out_channels,in_channels]  '''
        #shape = [kernel_size, kernel_size,  input_channels, filters] #that line was causing error  reference in [2]
        shape = [kernel_size, kernel_size, filters, input_channels]
        
        #print(shape)
        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.random.truncated_normal(shape, stddev=0.05), name='W')

        # Create new biases, one for each filter
        biases = tf.Variable(tf.constant(0.05, shape=[filters]), name='b')

        # TensorFlow operation for convolution
        #https://bit.ly/2rfsY6j
        layer = tf.nn.conv2d_transpose(input=input, filter=weights, output_shape=output_shape,
                                       strides=stride, padding=padding, name='nn.conv2d_transpose')
        #layer = tf.compat.v1.layers.conv2d_transpose(input, filters=weights, kernel_size=kernel_size, strides=stride, padding=padding)                               
        # Add the biases to the results of the convolution.
        layer += biases
        
        #tf.compat.v1.summary.histogram('weights', weights)
        #tf.compat.v1.summary.histogram('biases', biases)

        if activation == 'sigmoid':
            return tf.nn.sigmoid(layer, name='sigmoid')
        elif activation == None:
            return layer
        else:
            raise ValueError('The value of activation argument must be None or sigmoid as a string')


def PReLU(_x, alpha_value=0.1, name='PReLU', var_name='preluV', complx=False):
    """Parameters
      ----------
      **_x**: Input on which activation function will be applied.
        
      **alpha_value**: The value of aplha parameter, by default the value is 0.1. 
      Given value for aplha must be less than 1.
        
      **name**: Name for the scope, must be a string.
        
      **var_name**: Name for the variable scope of alpha
        
      **complx**: Cast output datatype to complex 128 if True. Default value is False
        
      Notes
      ----------
      Parametric Rectified Linear Unit.
      It follows:
      `f(x) = alpha * x for x < 0`,
      `f(x) = x for x >= 0`,
      where `alpha` is a learned array with the same shape as x.
      
      Input shape
      ----------
      Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model. Learn more about PReLU. 
      
      `Reference to the PReLu Algorithm. <https://medium.com/@shoray.goel/prelu-activation-e294bb21fefa>`_
      
      `Paper on PReLu Activation Function. <https://arxiv.org/abs/1502.01852>`_
        
      Output shape
      ----------
      Same shape as the input.
        
      Returns
      ----------
      **Returns tensor with same shape and type as input, just apply the activation function.**
      """
    with tf.name_scope(name):  
        alphas = tf.compat.v1.get_variable(var_name, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(alpha_value))
        act = tf.maximum(0.0, _x) + alphas * tf.minimum(0.0, _x)   
        if complx:
            return tf.dtypes.cast(act, dtype='complex128', name='ComplexCasting')
        elif not complx:
            return act
        else:
            raise ValueError('The value must be boolean i.e True or False')    

            
def CompileModel(labels, predictions, loss='mse', optimizer='adam', lr=1e-4, loss_name='MSE_Loss', opt_name='Adam_Optimizer'):
    """Parameters
       ----------
    
       **labels**: The data with output will be compared, in case Auto Encoder(AE) labels are same as inputs.
    
       **predictions**: Output of Model, or the output of Decoder in case of AE.
       
       **loss**: Name of loss, currently only `mse` is supported in future 'corssentrpoy' will be added also.
       
       **optimizer**: Name of optimizer, currently only `adam` is supported in future othres will be added also.
       
       **lr**: Learning Rate for the optimizer.
       
       **loss_name**: Name for the scope of loss function which is used in Tensorboard to create scope for loss.
       
       **opt_name**: Name for the scope of Optimizer which is used in Tensorboard to create scope for optimizer.
       
       Returns
       ---------- 
       **Returns the loss or cost and state of the optimizer or object to the optimizer.**"""
    #Implementing loss function   
    if loss=='mse':
        with tf.name_scope(loss_name):
            #cost = tf.compat.v1.losses.mean_squared_error(labels, predictions)
            cost = tf.reduce_mean(tf.square(labels - predictions)) #implementing MSE
            tf.compat.v1.summary.scalar('MSE_summary', cost)
    else: 
        raise ValueError('Invalid name for loss, currently only Mean Squared Error is supported!')
        
    #Implementing optimizer
    if optimizer=='adam':
        with tf.name_scope(opt_name):
            opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    else:
        raise ValueError('Invalid name for Optimizer, currently only Adam optimizer is supported!')
    return cost, opt

def GetAccuracy(labels, predictions, acc_name='accuracy'):
    """Parameters
       ----------
       
       labels: The data with output will be compared, in case Auto Encoder(AE) labels are same as inputs.
    
       predictions: Output of Model, or the output of Decoder in case of AE.
       
       acc_name: Name for the scope of Accurcay which is used in Tensorboard to create scope for accuracy.
       
       Returns
       ---------- 
       **Returns the accuracy for the given inputs.**
       """
    with tf.name_scope(acc_name):
        correct_prediction = tf.equal(predictions, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def WriteSummaries(cost, accuracy, input_images, k, snr, train_num):
    """Parameters
       ----------
       
       **cost**: The value of cost obtained from CompileModel() method.
    
       **accuracy**: The value of accuracy obtained from GetAccuracy() method.
       
       **input_images**: Input place holder for images.
       
       **k**: Number of pixels in compressed image, Channel dimension 'k' as the channel bandwidth which is computed from image dimensions.
       
       **snr**: The value of SNR in db scale, with this value the noise will be generated.
       
       **train_num**: This argument requires unique value each time model is run, to create different 
       directories for Tensorboard summaries to keep track of various runs of model. `str` type value
       
       Returns
       ---------- 
       **Returns the object to summaries for training, validation and merged_summary.**"""
    #Check whether the train_num argument is str type or not and if not change the dtype to str
    if type(train_num) != str:
        train_num = str(train_num)
    
    #Preparing setup for directories creation
    #Computing compression ratio 'k/n'
    lst = input_images.get_shape().as_list()
    lst.pop(0)
    n = np.prod(lst, dtype='float32') #number of pixels in input image.
    comp_ratio = k/n    
    #Creating directory for Train summary
    train_dir = os.path.join('Tensorboard','Compression_Ratio_'+str(int(comp_ratio*1000)/1000),'SNR_'+str(snr)+'dB',train_num,"TrainSummary")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    else:
        raise ValueError('Cannot Write Multiple Summaries in a single directory. Either delete previous directory or give new value to `train_num` argument. Refering to WriteSummaries method.')
    
    #Creating directory for validation summary
    valid_dir = os.path.join('Tensorboard','Compression_Ratio_'+str(int(comp_ratio*1000)/1000),'SNR_'+str(snr)+'dB',train_num,"ValidationSummary")
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    else:
        raise ValueError('Cannot Write Multiple Summaries in a single directory. Either delete previous directory or give new value to `train_num` argument. Refering to WriteSummaries method.')
    
    #Intializing File writers for summaries
    train_writer = tf.compat.v1.summary.FileWriter(train_dir+'/')
    valid_writer = tf.compat.v1.summary.FileWriter(valid_dir+'/')
    # Add the cost and accuracy to summary
    tf.compat.v1.summary.scalar('loss', cost)
    tf.compat.v1.summary.scalar('accuracy', accuracy)

    # Merge all summaries together
    merged_summary = tf.compat.v1.summary.merge_all()
    
    return train_writer, valid_writer, merged_summary

def TrainModel(x_train, x_test, placeholders, optimize, cost, accuracy, writers, batch_size=32, epochs=None, weights_name='JSCC', train_num='first', snr=20, early_stopping=False, es_count=10, k=None, c=None, train_str=None):
    """Parameters
       ----------
       
       **x_train**: Training data.
       
       **x_test**: Test data or validation data. 
       
       **placeholders**: A `list` containing the placeholders that is created to feed the input to model. 
       
       **optimize**: The object of optimizer which is returned by CompileModel() method. 
       
       **cost**: A tensor which is returned by CompileModel() method as a second return value.
       
       **accuracy**: A tensor which is returned by GetAccuracy() method.
       
       **writers**: A `list` containing three objects to the summaries in this order = [train_writer, valid_writer, merged_summary].
       
       **batch_size**: Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32. 
       
       **epochs**: Integer. Number of epochs to train the model. An epoch is an iteration over the entire data provided. 
       
       **weights_name**: Name for the weights files.
       
       **train_num**: This argument requires unique value each time model is run, to create different 
       directories for Tensorboard summaries to keep track of various runs of model. `str` type value.
       
       **snr**: The value of SNR in db scale, with this value the noise will be generated.
       
       **early_stopping**: A `bool` value, if set to `True` it will implement the early stopping. Default Value is `False`.
       `Learn more about early stopping <https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/>`_
       
       **es_count**: The number of epochs after which the training will be terminated if validation loss does not improve. 
       Default value is 10.
       
       **k**: Number of pixels in compressed image, Channel dimension 'k' as the channel bandwidth which is computed from image dimensions. 
       
       **c**: Number of filters in bottle neck layer.
       
       **train_str**: A `str` type value which will contain the information about training number if Auto training for all
       SNR values and compression ratios is implemented. Otherwise default value is `None`.
       
       Returns
       ----------
       **Returns a dictionary that contains the information about the model performance over each epoch.**"""
    #Check whether the writers argument is of list type or not.
    if type(writers) != list:
        raise ValueError('The value to the writers argument must be list in order given as: [train_writer, valid_writer, merged_summary]')
    #Check whether the train_num argument is str type or not and if not change the dtype to str
    if type(train_num) != str:
        train_num = str(train_num)
    
    #Assigning the value of summary writers from the list.
    train_writer = writers[0]
    valid_writer = writers[1]
    merged_summary = writers[2]
    #Assigning place holders
    input_images = placeholders[0] 
    P= placeholders[1] 
    snr_db=placeholders[2] 
    
    #Computing compression ratio 'k/n'
    lst = input_images.get_shape().as_list()
    lst.pop(0)
    n = np.prod(lst, dtype='float32') #number of pixels in input image.
    comp_ratio = k/n
    #check if the desired directory exist in the working directory, if does not exist creates the directory.
    train_dir = os.path.join('checkpoints','Compression_Ratio_'+str(int(comp_ratio*1000)/1000),'SNR_'+str(snr)+'dB',train_num)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    else:
        raise ValueError('Cannot Write Multiple Checkpoint files in a single directory. Either delete previous directory or give new value to `train_num` argument. Refering to TrainModel method.')
    
    
    #Creating an object to the batch maker class, which will provide the batches of data of given batch size while training. 
    train_data = BatchMaker(x_train, batch_size=batch_size)
    valid_data = BatchMaker(x_test, batch_size=batch_size)
    
    #creating the object for saving the model    
    saver = tf.compat.v1.train.Saver()
    init = tf.compat.v1.global_variables_initializer()
    #creating the session
    sess = tf.compat.v1.Session()
    #adding graph to the tensorboard
    train_writer.add_graph(sess.graph)

    #initializing variables
    sess.run(init)
    #creating a dictionary to store the model performance.
    history = {'Train_Accuracy':[], 'Train_Loss':[], 'Valid_Accuracy':[], 'Valid_Loss':[], 'Time':[], 'comp_ratio':[]}
    #Variables for early stopping
    prev_val_loss = math.inf #setting the value of prev_val_loss to almost infinity
    count = 0
    #creating the loop for training 
    if early_stopping:
        print('\n----- Early Stopping is implemented, Preparing to train the Model -----')
        print('System will train at compression ratio of {0:.3f}, and at SNR value of {1} dB\n\n'.format(comp_ratio, snr))
    else:
        print('\n----- Preparing to train the Model -----')
        print('System will train at compression ratio of {0:.3f}, and at SNR value of {1} dB\n\n'.format(comp_ratio, snr))
    #To track the Nan which apperas in the graph
    #check_op = tf.add_check_numerics_ops()
    for nb_epoch in range(1,epochs+1):
        ''' Training Part '''
        start_time = time.time() #keeps track of time
        #initializig variables for training results
        train_accuracy = 0.
        train_loss = 0.
        #creates the batch of given batch size as train_data is object of BatchMaker class
        i_batch=1
        for input_batch in train_data:
            st = time.time()
            # Run the optimizer using this batch of training data and also calculating the loss
            _, batch_loss = sess.run([optimize, cost], feed_dict={input_images:input_batch, P:[1], snr_db:[snr]})
            # Calculate the accuracy on the batch of training data
            
            train_accuracy += sess.run(accuracy, feed_dict={input_images:input_batch, P:[1], snr_db:[snr]})
            train_loss += batch_loss
            et = time.time()
            time_rem = int((train_data.batch_count-i_batch)*(et-st))
            print('\r' +'Epoch: ' + str(nb_epoch) + ' --> Training Batch No: '+ str(i_batch)+'/'+str(train_data.batch_count) + ' --> Train loss: {0:.5f}, ETA: {1}'.format(batch_loss, time_rem), end='')
            sys.stdout.flush() 
            i_batch += 1
        train_loss /= train_data.batch_count
        train_accuracy /= train_data.batch_count
        # Generate summary with the current batch of data and write to file
        train_sum = sess.run(merged_summary, feed_dict={input_images:input_batch, P:[1], snr_db:[snr]})
        train_writer.add_summary(train_sum, nb_epoch)
        
        ''' Validation Part '''
        #print('Training Completed, Now Performing Validation!')
        valid_accuracy = 0.
        valid_loss = 0.
        for valid_batch in valid_data:
            # Calculate the accuracy on the batch of validation data
            valid_accuracy += sess.run(accuracy, feed_dict={input_images:valid_batch, P:[1], snr_db:[snr]})
            # Calculate the loss on the batch of validation data
            valid_loss += sess.run(cost, feed_dict={input_images:valid_batch, P:[1], snr_db:[snr]})
        valid_loss /= valid_data.batch_count
        valid_accuracy /= valid_data.batch_count
        # Generate summary with the current batch of data and write to file
        valid_sum = sess.run(merged_summary, feed_dict={input_images:input_batch, P:[1], snr_db:[snr]})
        valid_writer.add_summary(valid_sum, nb_epoch)
        
        ''' Storing Model Performance'''
        history['Train_Accuracy'].append(train_accuracy)
        history['Train_Loss'].append(train_loss)
        history['Valid_Accuracy'].append(valid_accuracy)
        history['Valid_Loss'].append(valid_loss)
        history['comp_ratio'].append(int(comp_ratio))
        
        '''Giving Details about model performance'''
        end_time = time.time()
        history['Time'].append(int(end_time-start_time))
        print(' System Parameters at current training: SNR: {0} dB, No. of Filters: {1}, Compression ratio: {2:0.3f} '.format(snr, c, comp_ratio)+train_str)
        print("Epoch "+str(nb_epoch)+" completed : Time taken {0:.3f}".format(int(end_time-start_time)/60)+" minutes")
        Show_progress(nb_epoch, train_accuracy, train_loss, valid_accuracy, valid_loss)
        
        ''' Implementing Early Stopping '''
        if early_stopping:
            if valid_loss < prev_val_loss:
                print('Loss improved from: {0:.5f} to {1:.5f}, model saved to checkpoints directory.'.format(prev_val_loss, valid_loss))
                print('\t\t\t-------------------\n')
                prev_val_loss = valid_loss
                #saving the model
                saver.save(sess, './'+train_dir+'/'+weights_name)
                count = 0
            else:
                count += 1
                print('Loss did not improved, Early Stopping count: {0}/{1}'.format(count, es_count))
                print('\t\t\t-------------------\n')
                if count == es_count:
                    break 
        else:
            saver.save(sess, './'+train_dir+'/'+weights_name)
    #Stores the performance history of the model in a json file in the checkpoints directory 
    print('\n---------------------------------------------------------------------------')
    print('\nModel Training has been completed succesfully!')
    with open( './'+train_dir+'\\'+train_num+'_history.json', 'w') as fp:
        json.dump(history, fp)
    print('Perfomance history of the model has been saved to '+train_num+'_history.json '+'file in the respective checkpoints directory')
    return history