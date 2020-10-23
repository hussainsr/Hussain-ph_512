# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 16:32:34 2020

@author: hussa
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

def correlation(arr1,arr2):
	return np.fft.ifft(np.fft.fft(arr1)*np.conj(np.fft.fft(arr2)))

def shift_conv(arr, shift):
	# Takes and array 'arr'  and shifts it by 'shift' by cpmvolution with a delta function carrying 1 at location of shift
	k 		  = np.asarray([0 for i in range(len(arr))])
	k[shift]  = 1 # defining the delta function
	phase_k   = np.fft.fft(k)
	phase_arr = np.fft.fft(arr)
#	phase_arr = np.exp(2*np.pi*1J*k*shift/len(phase_arr))*phase_arr
	return np.fft.ifft(phase_arr*phase_k) # returning inv.fourier of the product of the two function's fourier transform

# Defining the array
x = np.linspace(-1,1,100)
y = np.exp(-x**2) # Gaussian (centered at x=0 with unit variance)
shift_rand = int(np.random.rand()*100)
y_shift = shift_conv(y,shift_rand) # Shifting the gaussian by random number
print('\n(Array size, shift)', [len(x),shift_rand])
corr = correlation(y_shift,y_shift)


plt.figure(figsize=(12,6))      
plt.title('Correlation of gaussian with itself (using fft)')
plt.xlabel('x')
plt.grid(alpha=0.75)
plt.ylabel('f(x)')
plt.plot(corr,label = 'Correlation result')
plt.legend()
plt.show()
plt.close()

# Tring out with normal and shifted gaussian
corr2 = correlation(y,y_shift)

plt.figure(figsize=(12,6))      
plt.title('Correlation of gaussian with itself (using fft)')
plt.xlabel('x')
plt.grid(alpha=0.75)
plt.ylabel('f(x)')
plt.plot(corr2,label = 'Correlation result')
plt.legend()
plt.show()
plt.close()