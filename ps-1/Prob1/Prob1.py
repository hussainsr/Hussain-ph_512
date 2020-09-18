# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:45:56 2020

@author: hussa
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate

# Question 1 ##################################################################

# Derivative scheme derived in part a
def deriv(fun,x,dx):
	return (8*fun(x+dx) - 8*fun(x-dx) + fun(x-2*dx) - fun(x+2*dx))/(12*dx) 

def function1(x):
	return np.exp(x)

def d_function1(x):     #derivative of function1
    return np.exp(x)

def function2(x):
	return np.exp(0.01*x)

def d_function2(x):		#derivative of function2
    return 0.01*np.exp(0.01*x)

# Calculates the difference between true and calculated derivative at a point
def error_estimate(fun,dfun,x,dx):
    Dx = deriv(fun, x, dx)
    error = np.abs(Dx-dfun(x))
#    print('Estimated derivative ',Dx)
#    print('Analytic derivative', np.exp(x), '\n')
#    print('Error', error)
    return error

##############################################################################

# checking funciton1
print('\n\nFor exp(x)')    
# Dependance of error on step size 
print(error_estimate(function1,d_function1,0.5,0.01)/error_estimate(function1,d_function1,0.5,0.005)) 
print('Above result shows that error scales with 4th-power of the step size coming from the leading error term derived in the notes\n\n')

# Error based on step size, minimum near machine precision ^ (1/5)
# error minimised at (10^(-16))**(1/5) ~ 10^(-3) 
error = []
for i in range(0,16):
	error.append(error_estimate(function1,d_function1,0.5,10**(-i))) 
	print('(dx, error) (',10**(-i),',',error[i],')')	
error = np.asarray(error)	
print('\nError is expected to be minimised at (10^(-16))**(1/5) ~ 10^(-3)')
plt.figure(figsize=(12,6))      
plt.title('Error vs stepsize (exp(x))')
plt.xlabel('Step-size')
plt.grid(alpha=0.75)
plt.ylabel('Error')    
plt.xscale('log')
plt.yscale('log')
plt.scatter([10**(-i) for i in range(len(error))], error) 
plt.show()   
plt.close()
print('\nError is minimum at dx = ',10**(-float(np.argmin(error))))    


# checking function2    
print('\n\n\nFor exp(0.01x)')  
# Dependance of error on step size   
print(error_estimate(function2,d_function2,100.0,2.0)/error_estimate(function2,d_function2,100.0,1.0)) # dependance of error on step size 
print('Same as abpve, result shows that error scales with 4th-power of the step size coming from the leading error term derived in the notes\n\n')

# Error based on step size, minimum near machine precision ^ (1/5)
# error minimised at (10^(-16))**(1/5)/(a*exp(0.01x)) ~ 10^(-3)/0.01 ~ 0.1
error = []
for i in range(16): # error based on step size, minimum near machine precision ^ (1/5)
	error.append(error_estimate(function2,d_function2,0.5,10**(-i))) 
	print('(dx, error) (',10**(-i),',',error[i],')')	
error = np.asarray(error)	
print('\nError is expected to be minimised at (10^(-16))**(1/5)/(0.01*exp(0.01x/5)) ~ 10^(-1)')
plt.figure(figsize=(12,6))      
plt.title('Error vs stepsize (exp(0.01x))')
plt.xlabel('Step-size')
plt.grid(alpha=0.75)
plt.ylabel('Error')    
plt.xscale('log')
plt.yscale('log')
plt.scatter([10**(-i) for i in range(len(error))], error) 
plt.show()   
plt.close()
print('\nError is minimum at dx = ',10**(-float(np.argmin(error))))    

