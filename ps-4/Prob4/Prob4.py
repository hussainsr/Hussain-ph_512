# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 18:06:41 2020

@author: hussa
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

def correlation(arr1,arr2):
	return np.fft.ifft(np.fft.fft(arr1)*np.conj(np.fft.fft(arr2)))

def convolution(arr1,arr2):
	return np.fft.ifft(np.fft.fft(arr1)*np.fft.fft(arr2))

def shift_conv(arr, shift):
	# Takes and array 'arr'  and shifts it by 'shift' by cpmvolution with a delta function carrying 1 at location of shift
	k 		  = np.asarray([0 for i in range(len(arr))])
	k[shift]  = 1 # defining the delta function
	phase_k   = np.fft.fft(k)
	phase_arr = np.fft.fft(arr)
	return np.fft.ifft(phase_arr*phase_k) # returning inv.fourier of the product of the two function's fourier transform

def pad(y):
	append_strip = np.asarray([0.0 for i in range(int(len(y)))]) # Padding with zeros, number equal to array size
	y = np.append(y,append_strip)
	return y

def fun3(x):
	y = np.zeros(len(x))
	y[0:len(y)//4] += 1.0
	return y

def fun4(x):
	y = np.zeros(len(x))
	for i in range(5):
		y[i*1000//4] += i+1. # Defining delta with increasing magnitube
	return y

x = np.linspace(0,5,1001)
y1 = fun3(x)
y2 = fun4(x)
conv = convolution(y1,y2) # Taking convolution with un-padded arrays

plt.figure(figsize=(8,6))      
plt.title('Functions')
plt.xlabel('x')
plt.grid(alpha=0.75)
plt.plot(x,y1,'.',label='f(x)')
plt.plot(x,y2,label='g(x)')
#plt.plot(corr,label = 'Correlation result')
plt.legend()
plt.show()
plt.close()

plt.figure(figsize=(8,6))      
plt.title('Convolution of functions')
plt.xlabel('x')
plt.grid(alpha=0.75)
plt.ylabel('f(x) o g(x)')
plt.plot(x,conv,label='Convoluted data')
#plt.plot(corr,label = 'Correlation result')
plt.legend()
plt.show()
plt.close()

#----------After padding with zeroes ----------------------------------------

y1 = pad(y1) # Padding with zeros, here using N zeros
y2 = pad(y2) # where N = len(x)
conv = convolution(y1,y2)

plt.figure(figsize=(8,6))      
plt.title('Functions')
plt.xlabel('x')
plt.grid(alpha=0.75)
#plt.ylabel('f(x)')
plt.plot(y1,'.',label='f(x)')
plt.plot(y2,label='g(x)')
#plt.plot(corr,label = 'Correlation result')
plt.legend()
plt.show()
plt.close()

plt.figure(figsize=(8,6))      
plt.title('Convolution of functions')
plt.xlabel('x')
plt.grid(alpha=0.75)
plt.ylabel('f(x) o g(x)')
plt.plot(conv[:1250],label='Convoluted data') # Plotting the truncated array (after trimming out the zeros)
#plt.plot(corr,label = 'Correlation result')
plt.legend()
plt.show()
plt.close()
