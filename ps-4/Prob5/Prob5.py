# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:31:31 2020

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

def convolution_rev(arr1,arr2):
	return np.fft.fft(np.fft.ifft(arr1)*np.fft.ifft(arr2))

def shift_conv(arr, shift):
	# Takes and array 'arr'  and shifts it by 'shift' by cpmvolution with a delta function carrying 1 at location of shift
	k 		  = np.asarray([0 for i in range(len(arr))])
	k[shift]  = 1 # defining the delta function
	phase_k   = np.fft.fft(k)
	phase_arr = np.fft.fft(arr)
#	phase_arr = np.exp(2*np.pi*1J*k*shift/len(phase_arr))*phase_arr
	return np.fft.ifft(phase_arr*phase_k) # returning inv.fourier of the product of the two function's fourier transform

def pad(y):
	append_strip = np.asarray([0.0 for i in range(int(2*len(y)))]) 
	y = np.append(y,append_strip)
	return y

def non_int_sine(x,l): # Return sine wave with 'l' periods in range of x
	return np.sin(2.*np.pi*l*x/(len(x)))

def sine_dft_analytic(x,l):
	N = len(x)+0. 
	k = np.fft.fftfreq(len(x),x[1]-x[0])*N # Reconfiguring the range of k to [-N/2,N/2-1]
	t1 = (1.-np.exp(-2.*np.pi*1J*(k-l)))/(1.-np.exp(-2.*np.pi*1J*(k-l)/N)) # term shifted by +l (corresponding to first exp term in sin)
	t2 = (1.-np.exp(-2.*np.pi*1J*(k+l)))/(1.-np.exp(-2.*np.pi*1J*(k+l)/N)) # term shifted by -l (corresponding to 2nd exp term in sin)
	dft_sine = (t1-t2)/(2J)
	return dft_sine 

def window(y):
	x = np.linspace(0,1,len(y))
	y1 = 0.5*(1.-np.cos(2.*np.pi*x)) # Hanning wind
	y_new = y1*y 
	return y_new # Returns product of input with the window

#----- Part c --------------------------------------------------------------

# Defining the input
l = 1.250 # Number of periods
x = np.linspace(0.0,999.0,1000) # Sampling thousand points 

y = non_int_sine(x,l) #

plt.figure(figsize=(8,6))      
plt.title('Non-integer sine wave')
plt.xlabel('x')
plt.grid(alpha=0.75)
plt.plot(y)
plt.show()
plt.close()

dft_sine_ana = sine_dft_analytic(x,l) # Analytic answer
dft_sine_fft = np.fft.fft(y)		  # Using fft

plt.figure(figsize=(8,6))      
plt.title('DFT of non-int sine function')
plt.yscale('log')
plt.xlabel('k')
plt.grid(alpha=0.75)
plt.plot(np.abs(dft_sine_fft),label='Calculated using fft')
plt.plot(np.abs(dft_sine_ana),'.',label='Analytic solution')
plt.legend()
plt.show()
plt.close()

print('Variance in dft results: ', np.std(np.abs(dft_sine_ana)-np.abs(dft_sine_fft)))

#----- Part d --------------------------------------------------------------

y_window = window(y)
dft_sine_window = np.fft.fft(y_window)

plt.figure(figsize=(8,6))      
plt.title('DFT of normal&windowed sine function')
plt.xlabel('k')
plt.grid(alpha=0.75)
#plt.ylabel('f(x)')
plt.yscale('log')
plt.plot(np.absolute(dft_sine_window[:]),label='Windowed DFT')
plt.plot(np.absolute(dft_sine_fft[:]),label='Normal DFT')
plt.legend()
plt.show()
plt.close()

#----- Part e --------------------------------------------------------------
N = len(x)

dft_window = np.fft.fft(0.5*(1-np.cos(2.*np.pi*x/len(x)))) # Taking DFT pf window function

plt.figure(figsize=(8,6))      
plt.title('DFT of window')
plt.xlabel('k')
plt.plot((dft_window),'.')
plt.show()
plt.close()

dft_sine_recon = ((dft_window[0])*np.roll(dft_sine_fft,0)+(dft_window[1])*np.roll(dft_sine_fft,1)+(dft_window[-1])*np.roll(dft_sine_fft,N-1))/N

plt.figure(figsize=(8,6))      
plt.title('DFT of windowed & recon.windowed sine function')
plt.xlabel('k')
plt.grid(alpha=0.75)
#plt.ylabel('f(x)')
plt.yscale('log')
plt.plot(np.absolute(dft_sine_window[:]),label='Windowed DFT')
#plt.plot(np.absolute(dft_sine_fft),label='Normal DFT')
plt.plot(np.absolute(dft_sine_recon[:]),label='Recon. window DFT')
plt.legend()
plt.show()
plt.close()

