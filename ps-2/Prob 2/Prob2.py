# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 22:27:45 2020

@author: hussa
"""
import numpy as np
import matplotlib.pyplot as plt

# Function log_2(x) with the argument scaled such that range (-1,1) maps to (0.5,1)
def f(x):
	return np.log((x+3)/4.)/np.log(2)

# Lorentzian for second example
def g(x):
	return np.exp(x)

# Rountine implementing chebyshev
def chebs(fun,x,d):
	y = fun(x)
	mat=np.zeros([len(x),d+1])
	Tn_2 = 1
	Tn_1 = x	
	mat[:,0]=Tn_2 		 # First term in chebyshev series
	mat[:,1]=Tn_1 		 # Second term in chebyshev series
	for i in range(2,d+1): # Assigning third term onwards 
		Tn = 2*x*Tn_1-Tn_2
		mat[:,i] = Tn
		Tn_2 = Tn_1
		Tn_1 = Tn
	pars=np.dot(np.linalg.pinv(mat),np.transpose(y))
	# recovering corresponding function values using the fitted coefficients
	return pars,mat

### Log_2(x) ########################

d = 50
n = 101
x = np.linspace(-1,1,n)
pars,mat = chebs(f,x,d)
y = np.zeros(n)
# Iterating through decreasing orders of chebyshev coefficients
print('\n\nUsing truncated Chebyshev for f(x)=log_2(x)')
for i in range(50):
	y = np.sum(pars[:d+1]*mat[:,:(d+1)],axis=1) # Truncating to order d
	if np.max(np.abs(y-f(x)))>10**(-6):
		print('Degree', d, ', For maximum error of 10^(-6)')
		print('i.e terms till T_',d,' are required for the fit')
		d+=1
		y = np.sum(pars[:d+1]*mat[:,:(d+1)],axis=1) 
		break
	else:
#		print('error is: ',np.max(np.abs(y-f(x))))
		d-=1

xx = x
# Fitting using legendre polynomials
coeff = np.polynomial.legendre.legfit(xx,f(xx),d)
yy = np.polynomial.legendre.legval(xx,coeff)
print('\nNumpy function for fitting legendre polynomials is also used to compare with Chebyshev results\n')

x = (x+3)/4. # Rescaling x for making plots with x-coord between (0.5,1)

plt.figure(figsize=(12,6))      
plt.title('f(x) = log_2(x)')
plt.ylabel('y')
plt.grid(alpha=0.75)
plt.xlabel('x')    
plt.scatter(x, y, marker='x') 
plt.scatter(x, yy) 
plt.plot(x, f(xx)) 
plt.legend(('Actual function','Chebyshev fit','Polynomial fit'),loc='upper right')
plt.show()   
plt.close()

plt.figure(figsize=(12,6))      
plt.title('Residuals from both fits')
plt.ylabel('Residuals')
plt.grid(alpha=0.75)
plt.xlabel('x')    
plt.yscale('log')
plt.scatter(x, np.abs(y-f(xx)),marker='o') 
plt.scatter(x, np.abs(yy-f(xx)),marker = 'x') 
plt.legend(('Chebyshev fit Residual','Polynomial fit residuals'),loc='upper right')
plt.show()   
plt.close()

print('Max errors: Chebyshev vs leg.polynomial \n(',np.max(y-f(xx)), '||' ,np.max(yy-f(xx)),')')
print('\nRMS errors: Chebyshev vs leg.polynomial \n(',np.sqrt(np.abs(np.mean(y**2-f(xx)**2)))
,'||',np.sqrt(np.abs(np.mean(yy**2-f(xx)**2)))
,')')

print('\nAbove results show, that for the same order')
print('Chebyshev does better in terms of minimizing maximum error (especially at the ends) by about a ')
print('factor of 2, while it doesnt do so well in RMS considering the error being 2 orders of')
print('magnitude higher which is apparent from the general trend seen in the residuals plot')
print('\nThis shows that Chebyshev is useful in terms of minimising maximum errors in places where overall accuracy isnt very important')