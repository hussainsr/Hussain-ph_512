# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 20:31:47 2020

@author: hussa
"""
import numpy as np
import matplotlib.pyplot as plt
import camb
import copy
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

def get_spectrum(pars,n,lmax=2000):
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    
    return tt[2:n+2]	# Getting rid of first to zero elements and resizing to data size

def chi_sq(data,y,err):
	# X,data are the given data, y is obtained from fit using 'pars'
	return np.sum((data-y)**2/err**2)

# Cooked-up function for returning y along with the derivatives wrt different paramters for newtons method
def f(params,n,dx):
	y=get_spectrum(params,n)
	derivs=np.zeros([n,len(params)])
	# Calculating derivative using central difference for getting rid of error from quadratic term in the taylor expansion
	for i in range(len(params)):		
		par = copy.deepcopy(params)
		par[i] += dx[i]
		y2=get_spectrum(par,n)
		par[i] -= 2*dx[i]
		y1=get_spectrum(par,n)
		derivs[:,i]=(y2-y1)/(2*dx[i])
	return y,derivs

# Function for performing newtons method
def newton(f,pars,delete_par,y,Ninv,dm,lmd):
	print('Initial par: ',pars)
	y_pred,derivs=f(pars,len(y),dm)
	derivs = np.delete(derivs, delete_par, axis=1)
	resid=y-y_pred #data minus current model
	rhs=derivs.T@(Ninv@resid)
	lhs=(derivs.T@Ninv@derivs)
	# Calculating the step to next iteration of the parameters
	step=np.linalg.inv(lhs*(np.eye(len(derivs[0]))*(1.0+lmd)))@rhs
	# For loop for getting rid of any parameter we wish to exclude from the analysis, say for e.g. Tau (can take multiple as well)
	for i in delete_par:
		step = np.insert(step, i, 0.0)
	m=pars+step
	print('Step ',step)
	print('Params: ',m)
	# returns the new paramters 'm', 'step' and the curvature matrix 'lhs'
	return m,step,lhs

# FUnction for runing Levenberg-Marquardt on top of Newtons
def LM(pars_cur,y_data,error_data,delete_index,f,lmd):
	n = len(y_data)
	N = np.zeros((n,n))
	np.fill_diagonal(N,error_data)
	Ninv=np.linalg.inv(N)
	y_cur = get_spectrum(pars_cur,n)
	chi_sq_curr = chi_sq(y_data,y_cur,error_data)
	dm = pars_cur/1000.0
	# lmd-> Lambda, for scaling the step size where required based on the chi-sq
	chi_sq_step = []
	lhs_cur = []
	# Loop for iterating and recalculating 
	for iter in range(1000):
		print('\n(Iteration, lambda)= ',[iter,lmd])
		pars_new,step,lhs = newton(f,pars_cur,delete_index,y_data,Ninv,dm,lmd)
		y_pred = get_spectrum(pars_new,n)
		chi_sq_new = chi_sq(y_data,y_pred,error_data)
		# Accepting if new chisquare is less than current
		if chi_sq_new < chi_sq_curr:
			chi_sq_step.append(chi_sq_new)
			pars_cur = pars_new
			lhs_cur = lhs
			y_cur = get_spectrum(pars_cur,n)
			chi_sq_curr = chi_sq_new
			# Taking sq-root of current lambda because of successful step
			lmd = lmd**0.5
			# Setting lambda to zero if it sees sufficient number of succesful steps
			if lmd < 1.1:
				lmd = 0.0
		# Decreasing the step size by scaling up the curvature matrix				
		else:
			print('Chi_sq- new:',chi_sq_new,' curr:',chi_sq_curr)
			if lmd == 0:	
				
				lmd += 2.0
			# Scaling lambda up by factor of 2 if this is not the first rejected step	
			else:
				lmd = lmd*2	
		# Terminating loop after all parameter steps drop below following tolerance				
		if np.sort(np.abs(step))[len(delete_index)] < 10**(-15):
			print('\n\nFinal parameter values: \n',pars_cur)
			print('Step at last iteration:\n',step)
			if len(lhs_cur)==0:
				lhs_cur = lhs
			break
	# Returning the parameters as they are towards the end, the curvature matrix, and chi-square for each step	
	return pars_cur, lhs_cur, chi_sq_step	

# Loading data
pars_ini=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
wmap=np.loadtxt('WMAP_data.txt')
error_data = wmap[:,2]
y_data = wmap[:,1]

#-- Part a -------------------------------------------------------------------

# Prep of initial conditions for parameters and other variables for Newton/LM Minimizer
pars_cur=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
delete_index = [3] # Parameter to be excluded - here tau
lmd = 0.0
pars_cur, lhs, chi = LM(pars_cur,y_data,error_data,delete_index,f,lmd)

# Error in the parameters can be calculated using the curvature matrix (lhs)		
delete_index = [] 	# Allowing all parameters
lmd = 10.0**16 	# Forcing LM to run only once
pars_cur_junk, lhs, chi_junk = LM(pars_cur,y_data,error_data,delete_index,f,lmd)
par_errs=np.sqrt(np.diag(np.linalg.inv(lhs)))
print('Error in parameters: \n',par_errs)	

'''
Final parameter values: 
 [7.08639614e+01 2.21988361e-02 1.07111939e-01 5.00000000e-02
 1.98762707e-09 9.70782297e-01]
Error in parameters: 
 [9.87916653e-02 3.04637834e-05 1.82646424e-04 1.40381365e-12
 7.49228460e-04]
'''


#-- Part b -------------------------------------------------------------------
		
# Running minimizer for only optical depth
delete_index = [0,1,2,4,5] # Removing everything except tau
lmd = 0.0
pars_cur2, lhs2, chi2 = LM(pars_cur,y_data,error_data,delete_index,f,lmd)

## Error in the parameters can be calculated using below		
par_errs=np.sqrt(np.diag(np.linalg.inv(lhs)))
print('Error in parameters: \n',par_errs)	# [0.00010476]
	

#-- Running all parameters from start
pars_cur=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
delete_index = []
pars_cur2, lhs2, chi2 = LM(pars_cur,y_data,error_data,delete_index,f)

# Error in the parameters can be calculated using below		
par_errs=np.sqrt(np.diag(np.linalg.inv(lhs2)))
print('Error in parameters: \n',par_errs)	

#Final par: [7.10138295e+01 2.22880134e-02 1.07300856e-01 5.06943875e-02 1.99554631e-09 9.73364402e-01]
#Error 	 : [1.01263327e-01 3.05068829e-05 1.88807242e-04 1.43347532e-12 7.54603795e-04]
'''
Final parameter values: 
 [7.10138295e+01 2.22880134e-02 1.07300856e-01 5.06943875e-02
 1.99554631e-09 9.73364402e-01]
Error in parameters: 
 [1.99574247e-01 4.49425676e-05 3.56331093e-04 6.65230882e-03
 2.57852753e-11 1.36318165e-03]
'''
