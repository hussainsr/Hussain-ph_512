# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:35:50 2020

@author: hussa
"""

import numpy as np


def f(x):
	return 1/(1+x**2)

def g(x):
	return np.cos(x)

def integrate_step(fun,x1,x2,tol):
	# Function used in class
    global function_counter
    x=np.linspace(x1,x2,5)
    y=fun(x)
    function_counter += 5
    area1=(x2-x1)*(y[0]+4*y[2]+y[4])/6
    area2=(x2-x1)*( y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
    myerr=np.abs(area1-area2)
    if myerr<tol:
        return area2
    else:
        xm=0.5*(x1+x2)
        a1=integrate_step(fun,x1,xm,tol/2)
        a2=integrate_step(fun,xm,x2,tol/2)
        return a1+a2

# My modified function
def single_call_integrate_step(*args):
	# Modified function for recycling already calculated f(x) values
	# Fewer arguments are passed for the first function call hence variable number of arguments used

	# Recovering variables required for calculation. 
	x1 = float(args[1])
	x2 = float(args[2])
	fun = args[0]
	tol = args[3]
	global function_counter
	
	# Code for checking if additional argument is recevied through recursive calls
	x=np.linspace(x1,x2,5)
	if len(args)>4:
		yy = args[4]  # Contains previously calculated f(x) value at x1, xm and x2
		y=np.asarray([yy[0],fun(x[1]),yy[1],fun(x[3]),yy[2]]) 
		# Only 2 new values calculated hence increment of only 2 to the counter
		function_counter += 2 
	else:		
		y=fun(x)
		# Only occurs at first function call, increment of 5 for calculating f(x) at all 5 x-values
		function_counter += 5

	area1=(x2-x1)*(y[0]+4*y[2]+y[4])/6
	area2=(x2-x1)*( y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
	myerr=np.abs(area1-area2)
	if myerr<tol:
		return area2
	else:
		xm=0.5*(x1+x2)
		a1=single_call_integrate_step(fun,x1,xm,tol/2,y[:3]) # Passing already calculated y-values as an additional argument
		a2=single_call_integrate_step(fun,xm,x2,tol/2,y[2:]) 
		return a1+a2

print('\n\n- Lorentzian ')
# Resetting counter and integrating using old function
function_counter = 0
ans=integrate_step(f,-10,10,0.001)
print('\n\nIntergrate step function calls = ',function_counter)
print('Integration result: ',ans)
count1 = function_counter


# Resetting counter and integrating using modified function
function_counter = 0
ans2=single_call_integrate_step(f,-10,10,0.001)
print('\nSingle_call Intergrate step function calls = ',function_counter)
print('Function calls saved: ', count1-function_counter)
print('Integration result: ',ans2)

print('\n\n-  Cosine')
# Resetting counter and integrating using old function
function_counter = 0
ans=integrate_step(g,-np.pi/2.0,np.pi/2.0,0.001)
print('\n\nIntergrate step function calls = ',function_counter)
print('Integration result: ',ans)
count1 = function_counter


# Resetting counter and integrating using modified function
function_counter = 0
ans2=single_call_integrate_step(g,-np.pi/2.0,np.pi/2.0,0.001)
print('\nSingle_call Intergrate step function calls = ',function_counter)
print('Function calls saved: ', count1-function_counter)
print('Integration result: ',ans2)

print('\n\n----Conclusion--------')
print('More than 50% reduction in function calls,')
print('to be exact, after deducting the initial steps requirment of 5,') 
print('the modified function reduces the number of calls by factor of (2/5) ')
print('such that for each value of x, f(x) is called just once')