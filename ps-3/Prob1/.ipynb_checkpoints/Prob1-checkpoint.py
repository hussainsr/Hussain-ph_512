# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 15:22:18 2020

@author: hussa
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

###################################################################################################

#-- Part (a) -------------------------------------------------------------------
# The general expression of paraboloid can be easily modeled using certain approximation
# Ideally z = ar^2 where the center is at origin which might not be the case always.
# Defining r^2 = x^2 + y^2
# Then z - z0 = a(x^2 + y^2 - 2x0*x - 2y0*y + (x0^2 + y0^2)) 
#	        z = (z0 + a(x0^2 + y0^2)) - 2ax0*x - 2ay0*y + ar^2 
# for R = r^2, we can define z in terms of linear parameters
# z = (z0 + a(x0^2 + y0^2)) - 2ax0*x - 2ay0*y + aR

f = open('dish_zenith.txt','r')
xyz = []
for t in f:
	t = np.asarray(t.strip().split(), dtype = 'float')
	xyz.append(t)
f.close()	

xyz = np.asarray(xyz)	
x = xyz[:,0]
y = xyz[:,1]
z = xyz[:,2]
R = x**2+y**2

# Defining the coefficient matrix
A = np.transpose([np.ones(len(z)),x,y,R])

#-- Part (b) -------------------------------------------------------------------
# Performing the fit
coeff = np.linalg.lstsq(A,z,rcond=None)[0]
# Comparing with our above definintion of the parameters
a = coeff[3]
y0 = -0.5*coeff[2]/a
x0 = -0.5*coeff[1]/a
z0 = coeff[0] - a*(x0**2+y0**2)

f = 1/(4*a)
par = [x0,y0,z0,a]
print('\nBest fit parameters:')
print('{x0,y0,z0,a} = ',par)

#-- Part (c) -------------------------------------------------------------------
stdz = abs(np.std(z-(a*((x-x0)**2+(y-y0)**2)+z0)))
stda = abs(stdz*a/z0)
stdf = stda*f/a
print('\nValue of focal length and estimted error: ',[f,stdf])
print('The error is well withing 1sigma of actual value(1500mm), error = ',abs(f-1500),' < ',stdf)

##--Bonus--###########################################################################################

# The above results give a good enough value for x0,y0,z0 but fails to find the direction of the primary axis
# The equation z = a*x'^2 + b*y'^2 captures the circular assymmetry 
# where x = cos(t)*x'+sin(t)*y' and y = cos(t)*y'-sin(t)*x'
# Hence we inverse rotate since we have x,y instaead of actual ones
# where x' = cos(t)*x-sin(t)*y and y' = cos(t)*y+sin(t)*x
# z = a(cos(t)*x-sin(t)*y)^2 + b(cos(t)*y+sin(t)*x)^2
# z = (ac^2+bs^2)*x^2+(as^2+bc^2)*y^2+2*(b-a)cs*(xy)
# z = (ac^2+bs^2)*X+(as^2+bc^2)*Y+2*(b-a)cs*W
# X = x^2, Y = y^2, W = xy

print('\n\nBonus results: ')

# First taking care of offsets using earlier results:
x = x-x0
y = y-y0
z = z-z0

X = x**2
Y = y**2
W = x*y

A = np.transpose([X,Y,W])
coeff2 = np.linalg.lstsq(A,z,rcond=None)[0]

# These coeffs can be used to calculate the angle from coordinate rotation and also the the principle axis paramters a,b
# Work-up shown in pdf included in the same folder

a1 = coeff2[0]
a2 = coeff2[1]
a3 = coeff2[2]
theta = 0.5*np.arctan(a3/(a2-a1))
a_new = 0.5*(a1+a2-a3/(2*np.cos(theta)*np.sin(theta)))
b_new = 0.5*(a1+a2+a3/(2*np.cos(theta)*np.sin(theta)))
print('\nAngle(degrees): ',theta*180/np.pi)
print('New focal lengths: f_a=',1/(4*a_new),' f_b=',1/(4*b_new))

##################################################################################################

'''   OLD CODE: This works if the data were  perfectly symmetric, i.e. x0 = average(x)
#-- Part (a) -------------------------------------------------------------------
# The general expression of paraboloid can be easily modeled using certain approximation
# Ideally z = ar^2 where the center is at origin which might not be the case always.
# We can make a transformation of the data such that x -> x + x0 and y -> y + y0
# where x0 and y0 are the mean (x,y) values
# Then z - z0 = a( x^2 + y^2) = ar^2, after this R = r^2 can give us a simple linear equation 
# z - z0 = aR

f = open('dish_zenith.txt','r')
xyz = []
for t in f:
	t = np.asarray(t.strip().split(), dtype = 'float')
	xyz.append(t)
f.close()	
	
xyz = np.asarray(xyz)	
# x0 and y0 for a good enough data would simply be the average values of x and y due to rotational symmetry
x0 = np.average(xyz[:,0])
y0 = np.average(xyz[:,1])
# z0 on the other hand would have to be the minimum values of z, in other words when x = x0 and y = y0
# we can temporaryliy calculate but since z is linear we can save it for later to be found using the linear fit
z0_temp = np.min(xyz[:,2])

# Performing transformation, shifting (x0,y0) to orgin
xyz2 = np.transpose([xyz[:,0]-x0,xyz[:,1]-y0,xyz[:,2]])
# Now converting x,y,z to R,z where R = r^2
R = xyz2[:,0]**2+xyz2[:,1]**2
R = np.transpose([R,np.ones(len(xyz2))])
z = xyz2[:,2]

#-- Part (b) -------------------------------------------------------------------
# At this point the data can be fitted with z - z0 = aR
a, z0 = np.linalg.lstsq(R,z,rcond=None)[0]
f = 1/(4*a)
par = [x0,y0,z0,a]
print('\nBest fit parameters:')
print('{x0,y0,z0,a} = ',par)

#-- Part (c) -------------------------------------------------------------------
stdz = abs(np.std(z-(a*R[:,0]+z0)))
stda = abs(stdz*a/z0)
stdf = stda*f/a
print('\nValue of focal length and estimted error: ',[f,stdf])
print('The error is well withing 1sigma of actual value(1500mm), error = ',f-1500,' < ',stdf)
'''
