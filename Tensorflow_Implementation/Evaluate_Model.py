# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:09:44 2019

@author: Danish
"""

import tensorflow as tf
from keras.datasets import cifar10
from JSCC_Methods import Normalize_pixels
from JSCC_visualization import  Get_PlotBySNR, Get_PlotByCompRatio, load_modelHistories
#from JSCC_visualization import load_model, get_pred, get_PSNR, read_json

################################# Preparing Inputs ############################
(trainX, _), (testX, _) = cifar10.load_data()
#normalizing the training and test data
x_train, x_test = Normalize_pixels(trainX, testX)
train_num='first'
weights_name = 'JSCC_test.meta'
f_name = 'first_history.json'

######################## Getting the single prediction #################
## Let us restore the saved model 
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#saver = load_model(sess, comp_ratio, snr, train_num, weights_name)
#test_img = x_test[0]
#pred_img = get_pred(test_img, snr, sess, pred_type='single')
#mse, psnr = get_PSNR(pred_img, test_img, data_len='single')
l =tf.trainable_variables()

######################### Obtain Plot # 1 ##########################
snr_lst = [0, 10, 20]
comp_ratios_lst = [0.032, 0.089, 0.154, 0.244, 0.325, 0.415, 0.488]
plt1_data = Get_PlotByCompRatio(x_test, snr_lst, comp_ratios_lst, train_num, weights_name)


######################### Obtain Plot # 2 ##########################
snr_test = [2,4,7,10,13,16,18,22,25,27]
snr_train_list = [1]
comp_ratio = 0.089 #compression ratio either 0.154 (1/6) or 0.089 (1/12)
plt2_data = Get_PlotBySNR(x_test, snr_test, snr_train_list, comp_ratio, train_num, weights_name, c_r='1/12')



########################## Load Model History ########################
#Get Training Histories for 1st plot trainings.
train_histories1 = load_modelHistories(snr_lst, comp_ratios_lst, train_num, f_name)

#Get Training Histories for 2nd plot trainings.
comp_ratios_lst = [0.089, 0.154]
train_histories2 = load_modelHistories(snr_train_list, comp_ratios_lst, train_num, f_name)

#Get Single history
#comp_ratio = 0.089
#snr = 1
#train_num = 'first'
#train_history = read_json(comp_ratio, snr, train_num)

        
#all_tensors = [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]
