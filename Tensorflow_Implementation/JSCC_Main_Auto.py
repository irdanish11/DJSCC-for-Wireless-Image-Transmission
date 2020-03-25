# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 00:57:21 2019

@author: Danish
"""

import numpy as np
from keras.datasets import cifar10
import tensorflow as tf
from JSSC_wrapper import conv2d_layer, conv2dTranspose_layer, PReLU, CompileModel, GetAccuracy, WriteSummaries, TrainModel
from JSCC_Methods import Normalize_pixels, NormalizationLayer, AWGN_layer, Calculate_filters

def JSC_auto(c, snr, train_str):          
    input_images = tf.compat.v1.placeholder(tf.float32, shape=[None, 32, 32,3], name='input_images')
    P = tf.compat.v1.placeholder(tf.float32, shape=[1], name='P')
    snr_db = tf.compat.v1.placeholder(tf.float32, shape=[1], name='SNR_db')
    
    ############################### Buliding Encoder ##############################
    ''' Correspondance of different arguments w.r.t to literature: filters = K, kernel_size = FxF, strides = S'''
    #1st Convoultional layer
    conv1 = conv2d_layer(input_images, input_channels=3, filters=16, kernel_size=5, stride=2, padding='VALID', name='CONV1') 
    prelu1 = PReLU(conv1, name='PReLu1', var_name='preluV1')
    #2nd Convoultional layer
    conv2 = conv2d_layer(prelu1, input_channels=16, filters=32, kernel_size=5, stride=2, padding='VALID', name='CONV2') 
    prelu2 = PReLU(conv2, name='PReLu2', var_name='preluV2')
    #3rd Convoultional layer
    conv3 = conv2d_layer(prelu2, input_channels=32, filters=32, kernel_size=5, stride=1, padding='SAME', name='CONV3') 
    prelu3 = PReLU(conv3, name='PReLu3', var_name='preluV3')
    #4th Convoultional layer
    conv4 = conv2d_layer(prelu3, input_channels=32, filters=32, kernel_size=5, stride=1, padding='SAME', name='CONV4') 
    prelu4 = PReLU(conv4, name='PReLu4', var_name='preluV4')
    #5th Convoultional layer
    conv5 = conv2d_layer(prelu4, input_channels=32, filters=c, kernel_size=5, stride=1, padding='SAME', name='CONV5')
    #˜z: output of the last convolutional layer
    z_tilta = PReLU(conv5, name='PReLu5', var_name='preluV5', complx=True) 
    
    ############################ Normalization Layer ##############################
    lst = z_tilta.get_shape().as_list()
    lst.pop(0)
    #computing channel dimension 'k' as the channel bandwidth.
    k = np.prod(lst, dtype='float32')
    
    #Implementing the normalization layer according to eq-1 in literature
    z = NormalizationLayer(z_tilta, k, P)
    
    #Adding noise to the signal, corrupted output of the communication channel represented as ẑ.
    z_cap = AWGN_layer(snr_db, z, lst, k, name='AWGN_layer')
    
    ############################### Building Decoder ##############################
    #extracting real part of the signal from ẑ.
    z_hat = tf.dtypes.cast(z_cap, dtype='float32', name='RealCasting')
    #1st Deconvolutional layer
    Dconv1 = conv2dTranspose_layer(z_hat, input_channels=c, filters=32, kernel_size=5, output_shape = tf.shape(conv4),
                                   stride=1, padding='SAME', name='CONVTranspose1') 
    prelu6 = PReLU(Dconv1, name='PReLu6', var_name='preluV6') 
    #2nd Deconvolutional layer
    Dconv2 = conv2dTranspose_layer(prelu6, input_channels=32, filters=32, kernel_size=5, output_shape = tf.shape(conv3),
                                   stride=1, padding='SAME', name='CONVTranspose2')  
    prelu7 = PReLU(Dconv2, name='PReLu7', var_name='preluV7')
    #3rd Deconvolutional layer
    Dconv3 = conv2dTranspose_layer(prelu7, input_channels=32, filters=32, kernel_size=5, output_shape = tf.shape(conv2),
                                   stride=1, padding='SAME', name='CONVTranspose3')  
    prelu8 = PReLU(Dconv3, name='PReLu8', var_name='preluV8')
    #4th Deconvolutional layer
    Dconv4 = conv2dTranspose_layer(prelu8, input_channels=32, filters=16, kernel_size=5, output_shape = tf.shape(conv1), 
                                   stride=2, padding='VALID', name='CONVTranspose4')  
    prelu9 = PReLU(Dconv4, name='PReLu9', var_name='preluV9')
    #5th Deconvolutional layer
    decoded = conv2dTranspose_layer(prelu9, input_channels=16, filters=3, kernel_size=5, output_shape = tf.shape(input_images), 
                                   stride=2, padding='VALID', name='CONVTranspose5', activation='sigmoid')  
    
    ################################ Compiling Model ##############################
    cost, opt = CompileModel(labels=input_images, predictions=decoded, loss='mse', optimizer='adam')
    
    accuracy = GetAccuracy(labels=input_images, predictions=decoded)
    #SNR value on which model will be trained.
    #snr=4
    train_writer, valid_writer, merged_summary = WriteSummaries(cost, accuracy, input_images, k, snr, train_num='tmp')
    
    ################################# Preparing Inputs ############################
    (trainX, _), (testX, _) = cifar10.load_data()
    #normalizing the training and test data
    x_train, x_test = Normalize_pixels(trainX, testX)
    
    writers = [train_writer, valid_writer, merged_summary]
    ph = [input_images, P, snr_db]
    
    ############################## Training the Model #############################
    history = TrainModel(x_train, x_test, placeholders=ph, optimize=opt, cost=cost, accuracy=accuracy, writers=writers, 
                         batch_size=16, epochs=5000, weights_name='JSCC_test', train_num='tmp', snr=snr, 
                         early_stopping=True, k=k,c=c, train_str=train_str)
    return history

########### Calculate NUmber of filters for Compression Ratio, k/n. ###########
   
#last val loss achieved at 0dB, comp ratio 0.155 was around 0.037****

########################### Main Scope #######################
if __name__ == '__main__':
    ''' For Plot # 1, use following values of SNR and compression ratios to train the model. '''
    #snr_val =[0, 10, 20] 
    #compression_ratios = [0.05, 0.09, 0.16, 0.25, 0.33, 0.42, 0.49]
    ''' For Plot # 2, use following values of SNR and compression ratios to train the model. '''
    snr_val =[19] 
    compression_ratios = [0.09]
    
    #computing filter size for given compression ratios
    filter_size = []
    for comp_ratio in compression_ratios:
        c = Calculate_filters(comp_ratio)
        filter_size.append(c)
      
    ############### Training Loop #############
    count=0  
    for snr in snr_val:
        for c in filter_size:
            count += 1
            print('\n___________________________________________________________________________')
            print('Reseting the graph!, Training Number: '+str(count)+'/'+str(len(snr_val)*len(filter_size)))
            train_str = 'Training Model Number: '+str(count)+'/'+str(len(snr_val)*len(filter_size))
            tf.compat.v1.reset_default_graph()
            print('Setting SNR value for Model at: '+str(snr)+' dB'+' Filter size will be: '+str(c))
            JSC_auto(c, snr, train_str)
            print('___________________________________________________________________________\n\n')

    
