# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:40:12 2019

@author: Danish
"""
import tensorflow as tf
import numpy as np
# scale pixels
def Normalize_pixels(train_data, test_data):
    #convert integer values to float
	train_norm = train_data.astype('float32')
	test_norm = test_data.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	return train_norm, test_norm
    
def DeNormalize_pixels(img):
    """Parameters
       ----------
       **img**: A single image or array of images, having pixel values between `0` and `1`.
       
       Returns
       ----------
       **Returns scaled image or array of images, having pixel values between `0` and `255`.** 
       """
    de_norm =img*255
    return de_norm.astype('uint8')

def Calculate_filters(comp_ratio, F=5, n=3072):
    """ Parameters
        ----------
        **comp_ratio**: Value of compression ratio i.e `k/n`
        
        **F**: Filter height/width both are same.
        
        **n** = Number of pixels in input image, calculated as `n = no_channels*img_height*img_width`
        
        Returns
        ----------
        **Number of filters required for the last Convolutional layer and first Transpose Convolutional layer for given compression ratio.**
        """
    K = (comp_ratio*n)/F**2
    return int(K)

def Show_progress(nb_epoch, train_accuracy, train_loss, valid_accuracy, valid_loss):
    #msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.5f}"
    msg = "----> Training Loss: {1:.7f},  Validation Loss: {2:.7f} <----"
    print(msg.format(nb_epoch, train_loss, valid_loss))

def NormalizationLayer(z_tilta, k, P, name='NormLayer'):
    """ Parameters
        ----------
        **z_tilta**: Value obtained from the last convolutional layer.
        
        **k**: Number of pixels in compressed image, Channel dimension `k` as the channel bandwidth which is computed from image dimensions.
            
        **P: The input placeholder P created for power.
            
        name: Name for the Normalization layer.
        
        Returns
        ----------
        **Returns the normalized signal/image.**"""
    with tf.name_scope(name):
        #calculating conjugate transpose of z_tilta
        z_conjugateT = tf.math.conj(tf.transpose(z_tilta, perm=[0,2,1,3], name='transpose'), name='z_ConjugateTrans')
        #Square root of k and P
        sqrt1 = tf.dtypes.cast(tf.math.sqrt(k*P, name='NormSqrt1'), dtype='complex128',name='ComplexCastingNorm')
        sqrt2 = tf.math.sqrt(z_conjugateT*z_tilta, name='NormSqrt2')#Square root of z_tilta* and z_tilta.
    
        div = tf.math.divide(z_tilta,sqrt2, name='NormDivision')
        #calculating channel input
        z = tf.math.multiply(sqrt1,div, name='Z')    
        return z    
    
def AWGN_layer(snr_db, z, lst, k, name='AWGN_layer'):
    """ Parameters
        ----------
        **snr_db**: Value of SNR in dB scale on which you want to train the model.: 
        
        **z**: Output signal/image from the normalization layer.
            
        **lst**: A list containig the dimensions of the given signal/image `z`.
            
        **k**: Channel dimension 'k' as the channel bandwidth which is computed from image dimensions.
            
        **name**: Name for the AWGN layer.
     
        Returns
        ----------
        **Returns the signal/image with AWGN noise added**
        
        See Also
        --------
        Reference to the AWGN noise algorithm implemented in this function: 
        `Learn more about AWGN noise. <https://www.gaussianwaves.com/2015/06/how-to-generate-awgn-noise-in-matlaboctave-without-using-in-built-awgn-function/>`_
        
        Reference to complex normal distribution: 
        `Learn more about Complex Normal Distribution. <https://en.wikipedia.org/wiki/Complex_normal_distribution>`_  
        
        Reference to Numpy implementation of complex normal distribution:
        `How to Implement Complex Normal Distribution in Numpy. <https://stackoverflow.com/a/55700398/6350067>`_ 
        """
    with tf.name_scope(name):
        #changing dtype to float64 as other values will have float64
        k=k.astype('float64')
        #Converting SNR from db scale to linear scale
        snr = 10**(snr_db/10.0)
        snr = tf.dtypes.cast(snr, dtype='float64', name='Float32_64Cast')
        ########### Calculating signal power ########### 
        #calculate absolute value of input
        abs_val = tf.math.abs(z, name='abs_val')
        #Compute Square of all values and after that perform summation
        summation = tf.math.reduce_sum(tf.math.square(abs_val, name='sq_awgn'), name='Summation')
        #Computing signal power, dividing summantion by total number of values/symbols in a signal.
        sig_pwr = tf.math.divide(summation,k, name='Signal_Pwr')
        #Computing Noise power by dividing signal power by SNR.
        noise_pwr = tf.math.divide(sig_pwr,snr, name='Noise_Pwr')
        #Computing sigma for noise by taking sqrt of noise power and divide by two because our system is complex.
        noise_sigma = tf.math.sqrt(noise_pwr/2, name='Noise_Sigma')
        #Casting/converting noise_sigma to complex data as our signal is of complex data type. Otherwise dtype mismatch occurs
        cmplx_noise_sigma = tf.dtypes.cast(noise_sigma, dtype='complex128', name='ComplexNoise_Sigma')
        #creating the complex normal distribution.
        cmplx_dist = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(lst[0],lst[1],lst[2]*2)).view(np.complex128)
        #Compute product of sigma and complex normal distribution
        noise = tf.math.multiply(cmplx_noise_sigma, cmplx_dist, name='Noise')
        #adding the awgn noise to the signal, noisy signal: ẑ
        z_cap = tf.math.add(z, noise, name='NoisySignal') 
        return z_cap
    
