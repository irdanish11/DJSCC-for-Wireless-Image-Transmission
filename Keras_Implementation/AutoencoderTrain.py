# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:39:40 2019

@author: Danish
1. Compute pwr combined (Real, Imag), (Extract R & I parts) Generate single distribution, Separate Sending, (R&I)
"""

from keras.datasets import cifar10
from AutoencoderModel import TrainAutoEncoder, Calculate_filters
import tensorflow as tf

(trainX, _), (testX, _) = cifar10.load_data()

def normalize_pixels(train_data, test_data):
    #convert integer values to float
	train_norm = train_data.astype('float32')
	test_norm = test_data.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	return train_norm, test_norm

#normalizing the training and test data
x_train, x_test = normalize_pixels(trainX, testX)
#compression_ratios = [0.06, 0.09, 0.17, 0.26, 0.34, 0.43, 0.49]
compression_ratios = [0.06, 0.09, 0.17]
SNR = [0, 10, 20]
for comp_ratio in compression_ratios:
    tf.keras.backend.clear_session()
    c = Calculate_filters(comp_ratio)
    print('---> System Will Train, Compression Ratio: '+str(comp_ratio)+'. <---')
    _ = TrainAutoEncoder(x_train, x_test, nb_epoch=750, comp_ratio=comp_ratio, batch_size=16, c=c, snr=10, saver_step=50)