# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 22:34:18 2020

@author: hussa
"""

import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
import os

# Code directly borrowed from the simple_reader
def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl

def read_file(filename):
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value
    gpsStart=meta['GPSstart'][()]
    #print meta.keys()
    #utc=meta['UTCstart'].value
    utc=meta['UTCstart'][()]
    #duration=meta['Duration'].value
    duration=meta['Duration'][()]
    #strain=dataFile['strain']['Strain'].value
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc

# ----------------------------------------------------------------------------
	
# Fucntion which is like 1-cos at edges but like box in middle
# Apparently it has a name :P Tukey window function
def window_func(n, t): 
	y = np.ones(n) # Defining a box extending the entire range
	m = int((1-t)*n/2)
	cos_edge = 0.5-0.5*np.cos(np.linspace(0,1,2*m)*2*np.pi)
	for i in range(m): # Smoothening out the transition at the edges by multiplying with cosine
		y[i] = y[i]*cos_edge[i]
		y[-(i+1)] = y[-(i+1)]*cos_edge[-(i+1)] 
	return y 

# Function for smoothening of power-spectrum by averaging using neighboring points
def smoothener(x, pix):
	t = np.zeros(len(x))
	t[:(pix+1)//2]  = 1 	# Box car
	t[-(pix+1)//2:] = 1
	t = t/pix 			# Normalising
	return np.fft.ifft(np.fft.fft(x,norm=None)*np.fft.fft(t,norm=None),len(x),norm=None)
	
# Function to calculate cross-corelation like the one in class
# Using the fourier method saves time because of fft being of time complexity N*log(N)
def cross_corr(x, y): 
	return np.fft.ifft(np.fft.fft(x,norm=None)*np.conj(np.fft.fft(y,norm=None)),norm=None)




# ---------------------------------------------------------------------------
# -- Main code ------------------------------------------------------------- #
# ---------------------------------------------------------------------------

# Note: please select the event you wish to use by commenting out the corresponding lines.

# Reading LIGO data
# Filedir gets the location where the python file is saved and sets to the 
# folder containing the h5py files	
filedir = os.getcwd()+'\\LOSC_Event_tutorial\\LOSC_Event_tutorial\\'

## Event 1
#fname1=filedir+'H-H1_LOSC_4_V2-1126259446-32.hdf5'
#fname2=filedir+'L-L1_LOSC_4_V2-1126259446-32.hdf5'
#fname3=filedir+'GW150914_4_template.hdf5'


## Event 2
#fname1=filedir+'H-H1_LOSC_4_V2-1128678884-32.hdf5'
#fname2=filedir+'L-L1_LOSC_4_V2-1128678884-32.hdf5'
#fname3=filedir+'LVT151012_4_template.hdf5'


## Event 3
#fname1=filedir+'H-H1_LOSC_4_V2-1135136334-32.hdf5'
#fname2=filedir+'L-L1_LOSC_4_V2-1135136334-32.hdf5'
#fname3=filedir+'GW151226_4_template.hdf5'


# Event 4
fname1=filedir+'H-H1_LOSC_4_V1-1167559920-32.hdf5'
fname2=filedir+'L-L1_LOSC_4_V1-1167559920-32.hdf5'
fname3=filedir+'GW170104_4_template.hdf5'


# ---------------------------------------------------------------------------
#   Matched filter ALGORITHM
# ---------------------------------------------------------------------------
	
print('reading file ',fname1)
strain_h,dt_h,utc_h=read_file(fname1)

print('reading file ',fname2)
strain_l,dt_l,utc_l=read_file(fname2)

time = np.arange(len(strain_h))*dt_h
#	freq = np.fft.fftfreq(len(time),dt_h)	
freq = np.arange(len(time))/32.	
#	freq = np.arange(len(time))/1.	

#th,tl=read_template('GW150914_4_template.hdf5')
template_name=fname3
th,tl=read_template(template_name)

#Checking for any direct detection like one done in class
plt.plot(cross_corr(strain_h,th));plt.show();plt.close()
plt.plot(cross_corr(strain_l,tl));plt.show();plt.close()

# 	PART A 
# ---------------------------------------------------------------------------

# There is obviously no clear indication from directly looking for correlation 

# Trying out windowing using a customized hanning window 
# which has a central flat region, like a box car with cosine edges
# 0.75 argument below is the fraction of the window occupied by the central flat part
wind = window_func(len(strain_h),0.75) 

plt.figure(figsize=(12,6))      
plt.title('Window function-Tukey')
plt.ylabel('f(x)')
plt.grid(alpha=0.75)
plt.xlabel('x')    
plt.plot(wind)
plt.show()
plt.close()

#wind = np.linspace(0,1,len(strain_h))*2*np.pi
#wind = 0.5*(1-np.cos(wind))

# Now strating off with windowing the data and template	
strain_h = strain_h*wind
strain_l = strain_l*wind
strain_h_fft1 = np.fft.fft(strain_h,norm=None)
strain_l_fft1 = np.fft.fft(strain_l,norm=None)

th = th*wind
tl = tl*wind
th_fft1 = np.fft.fft(th,norm=None)
tl_fft1 = np.fft.fft(tl,norm=None)

# Now I define the noise matrix assuming the noise in data is stationary
# This implies that the noise matrix in fourier space is diagonal/uncorrelated
# Hence I take the fft of the strain as my sigma and define N as its square
N_h	= np.abs(strain_h_fft1)**2
N_l	= np.abs(strain_l_fft1)**2

# Smoothening the power spectrum by averaging using neighbouring 4 points, 2 on either side 	
sN_h = smoothener(N_h,25) # Number needs to be an odd integer
sN_l = smoothener(N_l,25) # To evenly smooth either side

# Taking max of the original noise over smooth noise to recover peaks
smN_h = np.maximum(sN_h,N_h)		
smN_l = np.maximum(sN_l,N_l)		

th_fft = th_fft1/np.sqrt(sN_h)
tl_fft = tl_fft1/np.sqrt(sN_l)
strain_h_fft = strain_h_fft1/np.sqrt(sN_h)
strain_l_fft = strain_l_fft1/np.sqrt(sN_l)

plt.figure(figsize=(12,6))      
plt.title('Noise model (Livingston)')
#	plt.ylabel('Strain')
plt.grid(alpha=0.75)
plt.xlabel('Index/time')    
plt.loglog(freq,abs(N_l),label='noise')
plt.loglog(freq,abs(smN_l),label='smooth (max) noise')
plt.loglog(freq,abs(sN_l),label='smooth noise')
plt.legend()	
plt.show()
plt.close()

plt.figure(figsize=(12,6))      
plt.title('Whitened template/data')
#	plt.ylabel('Strain')
plt.grid(alpha=0.75)
plt.xlabel('Index/time')    
plt.loglog(freq,abs(th_fft),label='whitened template (Hanford)')
plt.loglog(freq,abs(strain_h_fft),label='whitened datat (Hanford)')
plt.legend()	
plt.show()
plt.close()

# Now that our noise model is ready we can use it to 


# 	PART B
# ---------------------------------------------------------------------------

mf_h_fft = np.conj(th_fft)*strain_h_fft
mf_l_fft = np.conj(tl_fft)*strain_l_fft
mf_h = np.fft.ifft(mf_h_fft,norm=None)
mf_l = np.fft.ifft(mf_l_fft,norm=None)

plt.figure(figsize=(12,6))      
plt.title('Matched filter')
plt.ylabel('Cross-correlation')
plt.grid(alpha=0.75)
plt.xlabel('Index/shift')    
#	plt.xscale('log')
plt.plot((mf_h)[:],label='Hanford')
plt.plot((mf_l)[:],label='Livington')
plt.legend()
plt.show();plt.close()
#		


# 	PART C 
# ---------------------------------------------------------------------------

# SNR from the results of matched filters of Hanford and livingston
#	print('argmax = ',np.argmax(abs(mf_h)),np.argmax(abs(mf_l)))
snr_l = np.abs(mf_l)/np.std(mf_l[np.argmax(abs(mf_l))+10:40000]) # Using a smaller chunck to calc. std to
snr_h = np.abs(mf_h)/np.std(mf_h[np.argmax(abs(mf_h))+10:40000]) # avoid the signal peak and window edge
snr = np.sqrt(np.max(snr_l)**2+np.max(snr_h)**2)
print('SNR Livingston: ', np.max(snr_l)) 
print('SNR Hanford '	, np.max(snr_h)) 
print('Combined event SNR:', snr)

# 	PART D 
# ---------------------------------------------------------------------------

#SNR from analytic expression of whitened template and data
th_w = np.fft.ifft(th_fft) # Windowed template
tl_w = np.fft.ifft(tl_fft) # Windowed template
strain_h_w = np.fft.ifft(strain_h_fft) # WIndowed strain
strain_l_w = np.fft.ifft(strain_l_fft) # WIndowed strain

snr_la = np.abs(cross_corr(tl_w,strain_l_w)/np.dot(tl_w,tl_w)**0.5)*(len(strain_h))**0.5
snr_ha = np.abs(cross_corr(th_w,strain_h_w)/np.dot(th_w,th_w)**0.5)*(len(strain_h))**0.5
snra = np.sqrt(np.max(snr_la)**2+np.max(snr_ha)**2)

plt.figure(figsize=(12,6))      
plt.title('Matched filter (Ana)')
plt.ylabel('Cross-correlation')
plt.grid(alpha=0.75)
plt.xlabel('Index/shift')    
#	plt.xscale('log')
plt.plot(snr_ha,label='Hanford')
plt.plot(snr_la,label='Livington')
plt.legend()
plt.show();plt.close()

print('SNR Livingston: ', np.max((snr_la))) 
print('SNR Hanford '	, np.max((snr_ha))) 
print('Combined event SNR:',snra)
	
# 	PART E 
# ---------------------------------------------------------------------------

# COmputing 50% weight frequency
psc_h = np.cumsum(np.abs(strain_h_fft1))/np.sum(np.abs(strain_h_fft1)) # PS cumulative livingston (fraction)
psc_l = np.cumsum(np.abs(strain_l_fft1))/np.sum(np.abs(strain_l_fft1))

tpsc_h = np.cumsum(np.abs(th_fft1))/np.sum(np.abs(th_fft1)) # template PS cumulative livingston (fraction)
tpsc_l = np.cumsum(np.abs(tl_fft1))/np.sum(np.abs(tl_fft1))

fh = freq[np.argwhere(psc_h>=0.5)[0][0]]
fl = freq[np.argwhere(psc_l>=0.5)[0][0]]
tfh = freq[np.argwhere(tpsc_h>=0.5)[0][0]]
tfl = freq[np.argwhere(tpsc_l>=0.5)[0][0]]

plt.figure(figsize=(12,6))      
plt.title('Cumulative sum PS')
plt.ylabel('Cum_sum')
plt.grid(alpha=0.75)
plt.xlabel('freq')    
#	plt.xscale('log')
plt.plot(freq,psc_h,label='Hanford')
plt.plot(freq,psc_l,label='Livington')
plt.legend()
plt.show();plt.close()

print('\nFreq. correcponding to half PS weight:')
print('Hanford: (Template,data)',tfh,fh)
print('Livingston: (Template,data)',tfl,fl)

# ---------------------------------------------------------------------------
# 	PART F 
# ---------------------------------------------------------------------------
	
# FOr determination of the angular postion range I use the difference in time of
# arrival and translate it to path difference based on the separation of the detectors

dist_detectors = 5000.0*10**3 # meters
speed_wave = 3.0*10**8
delta_t = np.abs(np.argmax(snr_h) - np.argmax(snr_l))*dt_h
# For a vertical angle theta, difference in arrival time time_a = dist_detectors*sin(theta)/speed
# We can get the time difference from the results of the matched fiter 
theta = np.arcsin(delta_t*speed_wave/dist_detectors)
#	delta_t = dist_detectors*cos(theta)/speed *d_theta
d_theta = 2*dt_h*speed_wave/(dist_detectors*np.cos(theta))

print('\nDIfference in arrival time: ', delta_t)

# Angular uncertainity is aruond 2degrees. This is not so good in terms positional
# uncertainity as the further we look for the event, worse the uncertanity get.
print('\nIn terms of error in position of event,\nfor distance x, uncertainity is rougly',d_theta,'*x\n\n')	
	


##--------------------------------------------------------------------------##

'''
filedir = os.getcwd()+'\\LOSC_Event_tutorial\\LOSC_Event_tutorial\\'

#fnames=glob.glob("[HL]-*.hdf5")
#fname=fnames[0]
fname=filedir+'H-H1_LOSC_4_V2-1126259446-32.hdf5'
print('reading file ',fname)
strain,dt,utc=read_file(fname)

#th,tl=read_template('GW150914_4_template.hdf5')
template_name=filedir+'GW150914_4_template.hdf5'
th,tl=read_template(template_name)


#spec,nu=measure_ps(strain,do_win=True,dt=dt,osamp=16)
#strain_white=noise_filter(strain,numpy.sqrt(spec),nu,nu_max=1600.,taper=5000)

#th_white=noise_filter(th,numpy.sqrt(spec),nu,nu_max=1600.,taper=5000)
#tl_white=noise_filter(tl,numpy.sqrt(spec),nu,nu_max=1600.,taper=5000)


#matched_filt_h=numpy.fft.ifft(numpy.fft.fft(strain_white)*numpy.conj(numpy.fft.fft(th_white)))
#matched_filt_l=numpy.fft.ifft(numpy.fft.fft(strain_white)*numpy.conj(numpy.fft.fft(tl_white)))

#copied from bash from class
# strain2=np.append(strain,np.flipud(strain[1:-1]))
# tobs=len(strain)*dt
# k_true=np.arange(len(myft))*dnu

### Add the bottom part in mf ###
# Assume noise is stationary, hence N is diag.
# Intersite time diff is around 10ms
# Smoothening needs to be around 3-4 points as the probablity that we have 
# successive low dice rolls is less

'''