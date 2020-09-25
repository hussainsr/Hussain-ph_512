# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:29:02 2020

@author: hussa
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib

# Halflife of all elements
global half_life, years, hours, minutes, microsecs

microsecs = 10**(-6)
minutes = 60.0
hours = 60.0*minutes
days = 24.0*hours
years = 365.25*days
half_life = np.asarray([4.468*10**9*years,
						24.1*days,
						6.7*hours,
						245500*years,
						75380*years,
						1600*years,
						3.8235*days,
						3.1*minutes,
						26.8*minutes,
						19.9*minutes,
						164.3*microsecs,
						22.3*years,
						5.015*years,
						138.376*days])

# Function for solving U238 decay chain
def fun(x,y):
	global half_life, years, hours, minutes, microsecs
	dydx=np.zeros(len(half_life)+1)	
	dydx[0]=-np.log(2)*y[0]/half_life[0]
	for i in range(1,len(half_life)):		
		dydx[i]=np.log(2)*y[i-1]/half_life[i-1]-np.log(2)*y[i]/half_life[i]
	dydx[-1]=np.log(2)*y[-2]/half_life[-1]
	return dydx

# Investigating the time evolution of U238 populations

print('\n\nPart a:')

y0=np.zeros(15)
y0[0]=1 
x0=half_life[0]*0.01
x1=half_life[0]*10     # Tracking till about 10 halflife period
#x1=20
ans_stiff=integrate.solve_ivp(fun,[x0,x1],y0,method='Radau')

print('\nFirst looking at the ratio of isotopes U238 and Pb206')
print('Here I use the integrate.solve_ivp from scipy because of its varying step size feature, in order to be able to handle isotope decay with very different half-life')
print('ranging from 4.5billion years(U238) to few microseconds(Po214)')

print('\n\nPart b\n')
print('Below is a plot of evolution of isotope population tracked over period of 10 half-life of U238')

matplotlib.rcParams.update({'font.size': 18})
plt.figure(figsize=(12,6))      
plt.title('Nuclear decay of U238 to Pb206')
plt.grid(alpha=0.75)
plt.xlabel('time (s)')    
plt.xscale('log')
plt.ylabel('Isotope population/fraction')
#plt.yscale('log')
plt.plot(ans_stiff.t, ans_stiff.y[0],marker = 'x') 
plt.plot(ans_stiff.t, ans_stiff.y[-1],marker = 'o') 
plt.legend(('U238 isotope population','Pb206 isotope population'),loc='center left')
plt.show()   
plt.close()

print('\nRatio of the above two species vs time')
matplotlib.rcParams.update({'font.size': 18})
plt.figure(figsize=(12,6))      
plt.title('Ratio of Pb206 to U238')
plt.grid(alpha=0.75)
plt.xlabel('time (s)')    
plt.xscale('log')
plt.ylabel('Isotope population/fraction')
#plt.yscale('log')
plt.plot(ans_stiff.t, ans_stiff.y[-1]/ans_stiff.y[0],marker = 'o') 
#plt.legend(('U238 isotope population','Pb206 isotope population'),loc='middle left')
plt.show()   
plt.close()

print('\n\nThe above plots show the decay of U238 to Pb206, which seems to follow a simple nuclear decay dynamic.')
print('Because the intermediate species have a halflife very small compared to the parent nuclei, in that time scale')
print('its almost as if U238 directly gets converted to Pb206')
print('hence the increase in the ratio when most of the U238 has decayed')
print('\n\nDue to the super-long life time of U238, it is impractical to compare its fraction to other intermediate')
print('isotopes because it would remain almost constant and we wouldnt be able make any determination from it.')
print('Rather its more interesting to consider U234 and Th230 due to their long halflife (but not too long)')
print('to probe for the age of rocks.')

y0=np.zeros(15)
y0[0]=1 
x0=1
x1=10**9
ans_stiff=integrate.solve_ivp(fun,[x0,x1],y0,method='Radau')

y0 = ans_stiff.y[:,-1]
x0 = x1
x1 = x0 + half_life[0]*20
ans_stiff=integrate.solve_ivp(fun,[x0,x1],y0,method='Radau')


matplotlib.rcParams.update({'font.size': 18})
plt.figure(figsize=(12,6))      
plt.title('Nuclear decay of U234 to Th230')
plt.grid(alpha=0.75)
plt.xlabel('time (s)')    
plt.xscale('log')
plt.ylabel('Isotope population/fraction')
#plt.yscale('log')
plt.plot(ans_stiff.t, ans_stiff.y[3],marker = 'x') 
plt.plot(ans_stiff.t, ans_stiff.y[4],marker = 'o') 
plt.legend(('U234 isotope population','Th230 isotope population'),loc='center left')
plt.show()   
plt.close()

print('\nRatio of the above two species vs time')
matplotlib.rcParams.update({'font.size': 18})
plt.figure(figsize=(12,6))      
plt.title('Ratio of U234 to Th230')
plt.grid(alpha=0.75)
plt.xlabel('time (s)')    
plt.xscale('log')
plt.ylabel('Isotope population/fraction')
#plt.yscale('log')
plt.plot(ans_stiff.t, ans_stiff.y[3]/ans_stiff.y[4],marker = 'o') 
#plt.legend(('U238 isotope population','Pb206 isotope population'),loc='middle left')
plt.show()   
plt.close()

print('\nThe above plot shows the decrease in the ratio as the U234 slowly decays to Th230 while Th230 is already in a steady state due to its shorter half-life')
print('The ratio of these isotopes found in rocks can be compared to give estimate of the age of the rock.')
print('Time on the x-axis matching the ration will thus be the age of the rock')