# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:49:18 2020

@author: hussa
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def fun1(x):
	return np.cos(x)

def fun2(x):
	return 1/(1+x**2)

# Function for performing polynomial interpolation
def polynm(x,xx,fun):
	y = fun(x)
	polyfit_coeff = np.polyfit(x,y,(len(x)-1))	
	return np.polyval(polyfit_coeff,xx)

# Function for performing spline interpolation
def spline(x,xx,fun):
	y = fun(x)
	spln=interpolate.splrep(x,y)
	return 	interpolate.splev(xx,spln)

# Function for performing rational polynomial interpolation
def rat(x,xx,fun,n,m):
	y = fun(x)
	mat=np.zeros([n+m-1,n+m-1])
	for i in range(n):
		mat[:,i]=x**i
	for i in range(1,m):
		mat[:,i-1+n]=-y*x**i
	pars=np.dot(np.linalg.inv(mat),y)
	p=pars[:n]
	q=pars[n:]
	print('\np = ',p,' q = ',q)
	top=0
	for i in range(len(p)):
		top=top+p[i]*xx**i
	bot=1
	for i in range(len(q)):
		bot=bot+q[i]*xx**(i+1)
	return top/bot	

# Small change in 'rat' to allow singular matrices
def rat_sing(x,xx,fun,n,m):
	y = fun(x)
	mat=np.zeros([n+m-1,n+m-1])
	for i in range(n):
		mat[:,i]=x**i
	for i in range(1,m):
		mat[:,i-1+n]=-y*x**i
	pars=np.dot(np.linalg.pinv(mat),y)
	p=pars[:n]
	q=pars[n:]
	print('\np = ',p)
	print('q = ',q)
	top=0
	for i in range(len(p)):
		top=top+p[i]*xx**i
	bot=1
	for i in range(len(q)):
		bot=bot+q[i]*xx**(i+1)
	return top/bot	

# Function for calling all methods
def interpolation_3types(x,xx,fun,n,m):
	y_poly = polynm(x,xx,fun)
	y_spline = spline(x,xx,fun)
	y_rat = rat(x,xx,fun,n,m)
	return y_poly, y_spline, y_rat

# Fixing the order rational fit with the general expansion of cosine function in mind
n = 8
m = 1
x = np.linspace(-np.pi/2.0,np.pi/2.0,n+m-1)  # x-values (given)
xx = np.linspace(-np.pi/2.0,np.pi/2.0,51) # x-values for interpolating	

print('\n\n\nFirst comparing the performance of the three interpolation methods for cosine fuunction')
y_poly,y_spline,y_rat = interpolation_3types(x,xx,fun1,n,m)
print('\nAs expected, for rational interp. the odd term coefficient are almost zero')

print('\nError in respective methods')
print('Maximum error \nPolynomial\t',np.max(y_poly-fun1(xx)))
print('Spline\t\t',np.max(y_spline-fun1(xx)))
print('Rational\t',np.max(y_rat-fun1(xx)),'\n')
print('Mean error \nPolynomial\t',np.average(y_poly-fun1(xx)))
print('Spline\t\t',np.average(y_spline-fun1(xx)))
print('Rational\t',np.average(y_rat-fun1(xx)),'\n')

print('\nBecause the function is well defined in terms of taylor series expansion the rational and polynomail fit seem to perform equally good and better compared to spline, especially near the edges')
plt.figure(figsize=(12,6))      
plt.title('Interpolation for cosine function')
plt.ylabel('y')
plt.grid(alpha=0.75)
plt.xlabel('x')    
plt.plot(xx, y_poly) 
plt.plot(xx, y_spline) 
plt.plot(xx, y_rat) 
plt.scatter(x, fun1(x)) 
plt.legend(('Polynomial', 'Spline', 'Rational', 'Data points'),loc='upper right')     
plt.show()   
plt.close()

plt.figure(figsize=(12,6))      
plt.title('Error/residuals')
plt.ylabel('y')
plt.grid(alpha=0.75)
plt.xlabel('x')    
plt.scatter(xx, y_poly-fun1(xx)) 
plt.scatter(xx, y_spline-fun1(xx)) 
plt.scatter(xx, y_rat-fun1(xx)) 
plt.plot(x, [0.0 for i in range(len(x))]) 
plt.legend(('Zero line','Polynomial', 'Spline', 'Rational'),loc='upper right')     
plt.show()   
plt.close()


print('\n\nNow for the lorentzian')
# Here the order of m is kept higher cause from the analytic expression of the function we know that the rational
# function will have denominator be 2nd order in x.
n = 1
m = 5
x = np.linspace(-1,1,n+m-1)
xx = np.linspace(-1,1,101)

print('\nHere the performance of rational function is expected to be exceptionally better because the Lorentzian can be exactly expressed in that form')

y_poly,y_spline,y_rat = interpolation_3types(x,xx,fun2,n,m)
print('\nIts clear from the nature of the coefficient that rational interpolation fits perfectly')

plt.figure(figsize=(12,6))      
plt.title('Interpolation for Lorenztain function')
plt.ylabel('y')
plt.grid(alpha=0.75)
plt.xlabel('x')    
plt.plot(xx, y_poly) 
plt.plot(xx, y_spline) 
plt.plot(xx, y_rat) 
plt.scatter(x, fun2(x)) 
plt.legend(('Polynomial', 'Spline', 'Rational', 'Data points'),loc='upper right')     
plt.show()   
plt.close()

plt.figure(figsize=(12,6))      
plt.title('Error/residuals')
plt.ylabel('y')
plt.grid(alpha=0.75)
plt.xlabel('x')    
plt.scatter(xx, y_poly-fun2(xx)) 
plt.scatter(xx, y_spline-fun2(xx)) 
plt.scatter(xx, y_rat-fun2(xx)) 
plt.plot(x, [0.0 for i in range(len(x))]) 
plt.legend(('Zero line','Polynomial', 'Spline', 'Rational'),loc='upper right')     
plt.show()   
plt.close()

print('\nError in respective methods')
print('Maximum error \nPolynomial ',np.max(y_poly-fun2(xx)))
print('Spline ',np.max(y_spline-fun2(xx)))
print('Rational ',np.max(y_rat-fun2(xx)),'\n')
print('Mean error \nPolynomial ',np.average(y_poly-fun2(xx)))
print('Spline ',np.average(y_spline-fun2(xx)))
print('Rational ',np.average(y_rat-fun2(xx)),'\n')

print('There results confirm that rational interpolation works best here, while the others are not so great')
print('\nOne issue with the current algo. is that the degree of the numerator and denominator have to be constrained otherwise leading to issue')
print('Let n=4, m=5 as suggested:')

n = 4
m = 5
x = np.linspace(-1,1,n+m-1)
xx = np.linspace(-1,1,101)
y_poly,y_spline,y_rat = interpolation_3types(x,xx,fun2,n,m)

print('\nClearly the order is all wrong')
plt.figure(figsize=(12,6))      
plt.title('Interpolation for Lorenztain function')
plt.ylabel('y')
plt.grid(alpha=0.75)
plt.xlabel('x')    
plt.plot(xx, y_rat) 
plt.scatter(x, fun2(x)) 
plt.legend(('Rational intrp','Data points'),loc='upper right')     
plt.show()   
plt.close()
print('\nThis problem can be dealt with by modyfying the method to allow for singular matrices')
print('Current method doesnt allow this and thus we dont find zeroes for the coefficient which we expect to be zero')

print('Follwing method takes care of this problem, now trying with n = 5, m = 10')
n = 5
m = 10
x = np.linspace(-1,1,n+m-1)
xx = np.linspace(-1,1,101)

y_rat = rat_sing(x,xx,fun2,n,m)
print('\nAnd now we got back what we initially expected from the coefficients')
plt.figure(figsize=(12,6))      
plt.title('Interpolation for Lorenztain function')
plt.ylabel('y')
plt.grid(alpha=0.75)
plt.xlabel('x')    
plt.plot(xx, y_rat) 
plt.scatter(x, fun2(x)) 
plt.legend(('Rational intrp','Data points'),loc='upper right')     
plt.show()   
plt.close()
print('\nALthough we can see like the case where we have incorrect interpolation we do have non-zero coeff for higher order, but they get suppressed because of the small step size giving back the good fit we had from the lower order result')