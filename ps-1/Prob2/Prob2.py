# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:47:57 2020

@author: hussa
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


# Question 2 ##################################################################

# Interpolation routine running using scipy-spline interpolation
def interpolation_routine(V,data):	
	spln=interpolate.splrep(data[:,0],data[:,1])
	K=interpolate.splev(V,spln)
	return K


# Reading data from lakeshore.txt file
data = []
with open('lakeshore.txt', 'r') as file:
    for t in file:
        row = t.strip().split()
        data.append([float(row[1]),float(row[0]),1/(float(row[2])*0.001)])    #factor of 0.001 to convert mV to V
# Column indexed zero contains data of Voltage and column indexed 1 correspoding temperature data
# data format is V, T, dT/dV
data = np.asarray(data)
# Sorting data wrt first column, i.e. V
data = data[np.argsort(data[:,0])] 


# Visualizing lakeshore data    
print('\n\n\nVisualizing lakeshore data')
plt.figure(figsize=(12,6))      
plt.title('Lakeshore data')
plt.ylabel('Temp (K)')
plt.grid(alpha=0.75)
plt.xlabel('Voltage (V)')    
plt.scatter(data[:,0], data[:,1]) 
plt.show()   
plt.close() 


#Performing interpolation on following voltages using scipy-spline interpolation
V = np.linspace(data[0,0],data[-1,0],201)    
K = interpolation_routine(V,data)


# Since the data point density is difficult to visualize along with interpolated point
# Use below code segment to zoom into a region, it basically truncate data to a given range
print('\nThe interpolation routine I wrote here uses scipy-spline interpolation')
print('\nShow below is a zoomed in section of the lakeshore data, its difficult to visualize the interpolation with the density of datapoints.')
print('Range of the x-axis can be modified according to need, for sample I am showing data between 1V and 1.2V')
data_cut = []
interp_cut  = []
range_low = 1.0
range_up  = 1.2
for i in range(len(V)):
	if V[i] < range_up and V[i] > range_low: 
		interp_cut.append([V[i],K[i]])
for i in range(len(data)):
	if data[i,0] < range_up and data[i,0] > range_low: 
		data_cut.append(data[i])
data_cut = np.asarray(data_cut)
interp_cut = np.asarray(interp_cut)

plt.figure(figsize=(12,6)) 
plt.title('Interpolated data (narrow-range)')
plt.ylabel('Temp (K)')
plt.grid(alpha=0.75)
plt.xlabel('Voltage (V)')    
plt.plot(interp_cut[:,0], interp_cut[:,1]) 
plt.scatter(data_cut[:,0], data_cut[:,1]) 
plt.legend(('Interpolated Line', 'Lakeshore Data'),loc='upper right')     
plt.show()   
plt.close()



# Estimating error

print('\nCommon method of testing models is to separate dataset into sample and test data and see how the model trained with sample data performs for the test data')
data_even = [] # separating alternate datapoints, even is sample data
data_odd = []  # test data
for i in range(len(data)-1): # '-1' in range is to leave out the last point as interpolation is done using even points as known
	if i%2 == 0:
		data_even.append(data[i])
	else:
		data_odd.append(data[i]) 
data_even = np.asarray(data_even)
data_odd = np.asarray(data_odd)
		
V = data_odd[:,0]
K = interpolation_routine(V,data_even)

error = np.abs(K-data_odd[:,1])  # error calc.

print('\nFollowing plot shows the error from comparing interpolated points on odd-datapoints (as test data) using even_datapoints (as sample data)')
print('Error is simply calculated as the difference of temperatures between odd-datapoints and interpolated points')
plt.figure(figsize=(12,6))      
plt.title('Interpolation error/residuals')
plt.ylabel('Error (dT)')
plt.grid(alpha=0.75)
plt.xlabel('Voltage (V)')
#plt.yscale('log')    
plt.plot(data_odd[:,0], [0.0 for i in range(len(data_odd))]) 
plt.scatter(data_odd[:,0], error) 
plt.legend(('Zero Line', 'Error data'),loc='upper right')     
plt.show()   
plt.close()

v_maxerror = data_odd[np.argmax(error),0]
print('Maximum error observed (in K) ', np.max(error), ' At V =', v_maxerror)
print('Average error observed (in K) ', np.average(error))		

print('\nWith the error obtained by interpolating over the odd-data points, we obtained a maximum difference of ~0.15K')
print('This dictates that the max error that the interpolation over all data would be under ~0.15K due to the smaller step size')