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
plt.ion()

def window_func(n, t): 
	y = np.ones(n)
	m = int((1-t)*n/2)
	cos_edge = 0.5-0.5*np.cos(np.linspace(0,1,2*m)*2*np.pi)
	for i in range(m):
		y[i] = y[i]*cos_edge[i]
		y[-(i+1)] = y[-(i+1)]*cos_edge[-(i+1)] 
	return y 

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
##--------------------------------------------------------------------------##

filedir = os.getcwd()+'\\LOSC_Event_tutorial\\LOSC_Event_tutorial\\'

fname=filedir+'H-H1_LOSC_4_V2-1126259446-32.hdf5'
print('reading file ',fname)
strain_h,dt_h,utc_h=read_file(fname)

fname=filedir+'L-L1_LOSC_4_V2-1126259446-32.hdf5'
print('reading file ',fname)
strain_l,dt_l,utc_l=read_file(fname)

#th,tl=read_template('GW150914_4_template.hdf5')
template_name=filedir+'GW150914_4_template.hdf5'
th,tl=read_template(template_name)

plt.figure(figsize=(12,6))      
plt.title('Hanford Data')
plt.ylabel('Strain')
plt.grid(alpha=0.75)
plt.xlabel('Index/time')    
#plt.legend(('Integarte.quad','My-integrator'),loc='upper right')     
plt.plot(strain_h)
#plt.savefig('Hanford_data.png')
plt.show()
plt.close()

plt.figure(figsize=(12,6))      
plt.title('Template Hanford')
plt.ylabel('Strain')
plt.grid(alpha=0.75)
plt.xlabel('Index/time')    
plt.plot(th[:67000])
plt.show()
plt.close()

plt.figure(figsize=(12,6))      
plt.title('FFT of Data Hanford')
plt.ylabel('Strain')
plt.grid(alpha=0.75)
plt.xlabel('Index/time')    
plt.plot(abs(np.fft.rfft(strain_h*window_func(len(strain_h),0.75)))[1000:1100],'.')
plt.plot(abs(np.fft.rfft(strain_h*window_func(len(strain_h),0.75)))[1000:1100])
#plt.savefig('Hanford_fft_data.png')
plt.show()
plt.close()

plt.figure(figsize=(12,6))      
plt.title('FFT of Template Hanford')
plt.ylabel('Strain')
plt.grid(alpha=0.75)
plt.xlabel('Index/time')    
plt.loglog(abs(np.fft.rfft(th[:])))
#plt.savefig('Hanford_template_fft.png')
plt.show()
plt.close()

########################################################
plt.figure(figsize=(12,6))      
plt.title('LIvingston Data')
plt.ylabel('Strain')
plt.grid(alpha=0.75)
plt.xlabel('Index/time')    
plt.plot(strain_l)
#plt.savefig('Liv_data.png')
plt.show()
plt.close()

plt.figure(figsize=(12,6))      
plt.title('Template LIvingston')
plt.ylabel('Strain')
plt.grid(alpha=0.75)
plt.xlabel('Index/time')    
plt.plot(tl[100:50000])
plt.show()
plt.close()

plt.figure(figsize=(12,6))      
plt.title('FFT of Data LIvingston')
plt.ylabel('Strain')
plt.grid(alpha=0.75)
plt.xlabel('Index/time')    
plt.loglog(abs(np.fft.rfft(strain_l)))
#plt.savefig('Liv_fft.png')
plt.show()
plt.close()

plt.figure(figsize=(12,6))      
plt.title('FFT of Template LIvingston')
plt.ylabel('Strain')
plt.grid(alpha=0.75)
plt.xlabel('Index/time')    
plt.loglog(abs(np.fft.fft(tl[100:60000])))
#plt.savefig('Liv_template_fft.png')
plt.show()
plt.close()
