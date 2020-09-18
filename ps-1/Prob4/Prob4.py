# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:50:10 2020

@author: hussa
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Question 4 #################################################################


# Equation for field due to ring of charge
def dE_field(z):
	# E = 2pi*r*z*sigma*dz/(4pi*epsilon*(z^2+r^2)**(3/2)
	# assuming sigma = 1/4piepisolon = 2*pi = 1 (ignoring constants)
	# z is position of ring, z0 is poistion of point
	r = (R**2-z**2)**0.5
	return r*(z0-z)/((z-z0)**2+r**2)**1.5


# Modified function to work with my integrator
def dE_field2(z,z0):
	# E = 2pi*r*z*sigma*dz/(4pi*epsilon*(z^2+r^2)**(3/2)
	# assuming sigma = 1/4piepisolon = 2*pi = 1 (ignoring constants)
	# z is position of ring, z0 is poistion of point
	r = (R**2-z**2)**0.5
	return r*(z0-z)/((z-z0)**2+r**2)**1.5


# Integrator operating using variable step via recursion
def integrate_step(fun,x1,x2,tol,dz,z0):
	x=np.linspace(x1,x2,5)
	if z0==R:
		return 0.0
	y=fun(x)
	area1=(x2-x1)*(y[0]+4*y[2]+y[4])/6
	area2=(x2-x1)*( y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
	myerr=np.abs(area1-area2)
	if myerr<tol or np.abs(x2-x1) < dz:  # Added additional clause to prevent infinte recursion
		return area2
	else:
		xm=0.5*(x1+x2)
		a1=integrate_step(fun,x1,xm,tol/2,dz,z0)
		a2=integrate_step(fun,xm,x2,tol/2,dz,z0)
		return a1+a2

# Small fix for singularity
def integrate_step2(fun,x1,x2,tol,dz,z0):
	if z0==R: # Averages neighbourhood points
		return (integrate_step2(fun,x1,x2,tol,dz,z0+dz) + integrate_step2(fun,x1,x2,tol,dz,z0-dz))/2.0
	x=np.linspace(x1,x2,5)
	y=fun(x,z0)
	area1=(x2-x1)*(y[0]+4*y[2]+y[4])/6
	area2=(x2-x1)*( y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
	myerr=np.abs(area1-area2)
	if myerr<tol or np.abs(x2-x1) < dz:  # Added additional clause to prevent infinte recursion
		return area2
	else:
		xm=0.5*(x1+x2)
		a1=integrate_step2(fun,x1,xm,tol/2,dz,z0)
		a2=integrate_step2(fun,xm,x2,tol/2,dz,z0)
		return a1+a2
	
	
#Let
print('\n\nDefining the variable for the problem')
R = 1.0
dz = 0.00001    # Choice is a bit arbitrary, small enough to converge to a good enough value
tol = 0.00001
z = 4.0
n = 100
print('Radius of sphere = ',R)		
print('Min. step size   = ',dz)		
print('Tolerance        = ',tol,'\n\n')		


E_z2 = []
E_z3 = []
z0 = 0.0
for i in range(n+1):
	z0 = i*z/(n)
	E_z2.append(integrate.quad(dE_field,-R,R))		 
	E_z3.append(integrate_step(dE_field,-R,R,tol,dz,z0))
E_z2 = np.asarray(E_z2)
E_z2 = E_z2[:,0]
z_points = np.linspace(0,z0,n+1)

print('A run-time error occurs when we reach z=1.0=R where the denominator part of the function we define becomes zero.')
print('Currently the integrator is handeling the singularity through a condition that makes it return zero when encountered while quad seems to work without any issue.')
print('[Note: Only orange dots are visible because the blue ones got overlapped')
plt.figure(figsize=(12,6))      
plt.title('E_z')
plt.ylabel('Electric field')
plt.grid(alpha=0.75)
plt.xlabel('z-axis')    
plt.scatter(z_points, E_z3) 
plt.scatter(z_points, E_z2) 
plt.legend(('My-integrator','Integarte.quad'),loc='upper right')     
plt.show()   
plt.close()

print('\nBelow is plot with only points from my-integrator visible with quad results appearing as line')

plt.figure(figsize=(12,6))      
plt.title('E_z')
plt.ylabel('Electric field')
plt.grid(alpha=0.75)
plt.xlabel('z-axis')    
plt.scatter(z_points, E_z3) 
plt.plot(z_points, E_z2) 
plt.legend(('Integarte.quad','My-integrator'),loc='upper right')     
plt.show()   
plt.close()

print('\nOne simple way to handle the singularity would be to analyse in neighbourhood of the point and take the average of it')

E_z3 = []
z0 = 0.0
for i in range(n+1):
	z0 = i*z/(n)
	E_z3.append(integrate_step2(dE_field2,-R,R,tol,dz,z0)) #calling integrate_step2 that contains the fix

plt.figure(figsize=(12,6))      
plt.title('E_z')
plt.ylabel('Electric field')
plt.grid(alpha=0.75)
plt.xlabel('z-axis')    
plt.scatter(z_points, E_z3) 
plt.plot(z_points, E_z2) 
plt.legend(('Integarte.quad','My-integrator'),loc='upper right')     
plt.show()   
plt.close()

print('\nConsidering the simpicity, this seems to have worked pretty well, so now this integrator also can take care of such points')