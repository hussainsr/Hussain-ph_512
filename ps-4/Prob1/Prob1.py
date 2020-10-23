# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 13:15:26 2020

@author: hussa
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams.update({'font.size': 18})

def shift_conv(arr, shift):
	# Takes and array 'arr'  and shifts it by 'shift' by convolution with a delta function carrying 1 at location of shift
	k 		= np.asarray([0 for i in range(len(arr))])
	k[shift]  = 1. # defining the delta function
	phase_k   = np.fft.fft(k)
	phase_arr = np.fft.fft(arr)
#	phase_arr = np.exp(2*np.pi*1J*k*shift/len(phase_arr))*phase_arr
	return np.fft.ifft(phase_arr*phase_k) # returning inv.fourier of the product of the two function's fourier transform

x 		= np.linspace(-1,1,100)
y 		= np.exp(-x**2) 		  # Unit mean, variance gaussian
y_shift = shift_conv(y,len(y)//4) # Shifting the gaussian by quarter of array size
print('\n(Length of array, shift): ',[100,100/4])

plt.figure(figsize=(12,6))      
plt.title('Array shift')
plt.xlabel('x')
plt.grid(alpha=0.75)
plt.ylabel('f(x)')
plt.plot(x,y,label='original gaussian')
plt.plot(x,y_shift,label='shifted gaussian')
plt.legend()
plt.show()
plt.close()

