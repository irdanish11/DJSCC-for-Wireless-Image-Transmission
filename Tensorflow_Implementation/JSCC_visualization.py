# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 00:03:39 2019

@author: Danish
"""
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import json

def load_model(sess, comp_ratio, snr, train_num, weights_name):
    """ Action:
        ----------
        Loads a saved model via saver.restore().
        
        Arguments:
        ----------
        **sess**: Object to the Tensorflow session which is created.
            
        **comp_ratio**: Specific compression ratio, for which you want to load the model. See checkpoints directory for 
        available compression ratios.
            
        **snr**: Specific SNR, for which you want to load the model. See checkpoints directory for available SNR val. 
            
        **train_num**: The name of the training the same 'name' which was given to `train_num` argument in JSSC_wrapper.TrainModel()
        while training the model for given SNR, and compression ratios. 
            
        **weights_name**: Name of the weights file, must be `str` type and contain the `.meta` extension. E.g `JSCC_test.meta`
        
        Returns:
        ----------
        **Loads the model weights in the current graph which is created and returns a saver object**.
        """
    #managing the path for loading the model.    
    path = './checkpoints/Compression_Ratio_'+str(comp_ratio)+'/SNR_{0}dB/'.format(snr)+train_num+'/'
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.compat.v1.train.import_meta_graph(path+weights_name)
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint(path))
    return saver  

def get_pred(test_img, snr, sess, pred_type):
    """Arguments:
       ----------
       **test_img**: Single image or multiple images. But input must have four dimensions. Dimensions for single image 
       [1, height, width, channels] and for multiple images [X, height, width, channels].
           
       **snr**: Value of SNR for the channel, this value will be used to obtain the channel for this particular value of
       SNR,i.e signal/image will pass through the channel which will have this particular value of SNR.
           
       **sess**: Object of the tensorflow session which is created. 
         
       **pred_type**: A `str` type value, `single` if only one image prediction is desired, `multiple` if input contains
       multiple images or array of images.
           
       Returns:
       ----------
       **Returns the decoded image/images, which are output of the autoencoder.**
       """
    graph = tf.compat.v1.get_default_graph()
    input_images = graph.get_tensor_by_name("input_images:0")
    P = graph.get_tensor_by_name('P:0' )
    snr_db = graph.get_tensor_by_name('SNR_db:0' )
    decoded = graph.get_tensor_by_name('CONVTranspose5/sigmoid:0')
    if pred_type == 'single':
        pred = sess.run(decoded, feed_dict={input_images:[test_img], P:[1], snr_db:[snr]})[0]
        return pred 
    elif pred_type == 'multiple': 
        pred = sess.run(decoded, feed_dict={input_images:test_img, P:[1], snr_db:[snr]})
        return pred
    else:
        raise ValueError('Invalid value for `pred_type`, it must be `single` if you want to predict single image, and `multiple` if you want to predict batch or mutiple images.')

def get_PSNR(pred_img, test_img, data_len, N=255):
    if data_len == 'single':
        mse = np.sum(np.square(pred_img-test_img))
        psnr = 10*np.log10(N**2/mse)
        return mse, psnr
    elif data_len == 'multiple':
        mse = np.sum(np.square(pred_img-test_img))/len(pred_img)
        psnr = 10*np.log10(N**2/mse)
        return mse, psnr
    else:
        raise ValueError('Invalid value for `data_len`, it must be `single` if given data contains single image, and `multiple` if given data contains batch or mutiple images.')

def PlotsData_SNRtest(x_test, snr_test, snr_train, comp_ratio, train_num, weights_name):
    """Give the data for the plots on the basis of different SNR_test
    
       **snr_test**: A `list` containing different values of SNR on which you want to test the model which was
       trained on the given `train_snr` value."""
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = load_model(sess, comp_ratio, snr_train, train_num, weights_name)
    data = {'Train_snr':[snr_train], 'Test_snr':[], 'PSNR':[], 'MSE':[]}
    for snr in snr_test:
        print('\n\r' +'Now preparing plot data for SNR Test: {0}'.format(snr)+' dB'  , end='')
        sys.stdout.flush() 
        pred_img_batch = get_pred(x_test, snr, sess, pred_type='multiple')
        mse_batch, psnr_batch = get_PSNR(pred_img_batch, x_test, data_len='multiple') 
        data['Test_snr'].append(snr)
        data['PSNR'].append(psnr_batch)
        data['MSE'].append(mse_batch)
    sess.close()
    return data

def plot_model(x, y, color, marker, title, x_lablel, y_label):
    plt.plot(x, y, c = color, marker = marker)
    plt.title(title)
    plt.xlabel(x_lablel)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()
    
def Get_PlotBySNR(x_test, snr_test, snr_train_list, comp_ratio, train_num, weights_name, c_r):
    markers = ["*", "s", "o", "X", "d", "v"]
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    title = 'AWGN channel (k/n = {0})'.format(c_r)
    x_lablel = 'SNR_test (dB)'
    y_label = 'PSNR (dB)'
    count = 0
    history = []
    for snr in snr_train_list:
        tf.compat.v1.reset_default_graph()
        print('\n\n---> Plotting Performance of Model for {0} dB SNR <---'.format(snr))
        data = PlotsData_SNRtest(x_test, snr_test, snr, comp_ratio, train_num, weights_name)
        x = data['Test_snr']
        y = data['PSNR']
        plot_model(x, y, colors[count], markers[count], title, x_lablel, y_label)
        count += 1
        history.append(data)
    return history

def PlotsData_CompRatio(x_test, snr_train, comp_ratios_lst, train_num, weights_name):
    """Give the data for the plots on the basis of different Compression Ratios
    
       **comp_ratios_lst**: A `list` containing different values of Compression ratios on which you want to test 
       the model which was trained on the given `train_snr` value and each of the given Compression ratios."""
    data = {'Train_snr':[snr_train], 'Compression_ratio':[], 'PSNR':[], 'MSE':[]}
    for comp_ratio in comp_ratios_lst:
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = load_model(sess, comp_ratio, snr_train, train_num, weights_name)
        print('\r' +'Now preparing plot data for Compression Ratio: {0:.3f}'.format(comp_ratio)+' dB\n\n'  , end='')
        sys.stdout.flush() 
        pred_img_batch = get_pred(x_test, snr_train, sess, pred_type='multiple')
        mse_batch, psnr_batch = get_PSNR(pred_img_batch, x_test, data_len='multiple') 
        data['Compression_ratio'].append(comp_ratio)
        data['PSNR'].append(psnr_batch)
        data['MSE'].append(mse_batch)
    sess.close()
    return data

def Get_PlotByCompRatio(x_test, snr_lst, comp_ratios_lst, train_num, weights_name):
    markers = ["*", "s", "o", "X", "d", "v"]
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    title = 'AWGN channel'
    x_lablel = 'Compression Ratio (k/n)'
    y_label = 'PSNR (dB)'
    count = 0
    history = []
    for snr_train in snr_lst:
        tf.compat.v1.reset_default_graph()
        print('\n\n-----> Plotting Performance of Model for {0} dB SNR <-----\n'.format(snr_train))
        data = PlotsData_CompRatio(x_test, snr_train, comp_ratios_lst, train_num, weights_name)
        x = data['Compression_ratio']
        y = data['PSNR']
        plot_model(x, y, colors[count], markers[count], title, x_lablel, y_label)
        count += 1
        history.append(data)
    return history

def read_json(comp_ratio, snr, train_num, f_name='first_history.json'):
    path = './checkpoints/Compression_Ratio_'+str(comp_ratio)+'/SNR_{0}dB/'.format(snr)+train_num+'/'+f_name
    with open(path, 'r') as f:
        dictionary = json.load(f)
    return dictionary

def load_modelHistories(snr_lst, comp_ratios_lst, train_num, f_name='first_history.json'):
    data = []
    for snr in snr_lst:
        data_compR = []
        for comp_ratio in comp_ratios_lst:
            data_compR.append(read_json(comp_ratio, snr, train_num))
        data.append(data_compR)