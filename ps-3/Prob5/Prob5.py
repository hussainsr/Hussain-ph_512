# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:28:25 2020

@author: hussa
"""

import numpy as np
import matplotlib.pyplot as plt
import camb
import copy
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

def get_spectrum(pars,n,lmax=2000):
    #print('pars are ',pars)
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

def newton(f,pars,delete_par,y,Ninv,dm,lmd):
	print('Initial par: ',pars)
	y_pred,derivs=f(pars,len(y),dm)
	derivs = np.delete(derivs, delete_par, axis=1)
	resid=y-y_pred #data minus current model
	rhs=derivs.T@(Ninv@resid)
	lhs=(derivs.T@Ninv@derivs)
	step=np.linalg.inv(lhs*(np.eye(len(derivs[0]))*(1.0+lmd)))@rhs
	for i in delete_par:
		step = np.insert(step, i, 0.0)
	m=pars+step
	print('Step ',step)
	print('New Params: ',m)
	return m,step,lhs


def LM(pars_cur,y_data,error_data,delete_index,f):

	n = len(y_data)
	N = np.zeros((n,n))
	np.fill_diagonal(N,error_data)
	Ninv=np.linalg.inv(N)
	y_cur = get_spectrum(pars_cur,n)
	chi_sq_curr = chi_sq(y_data,y_cur,error_data)
	dm = pars_cur/1000.0
	lmd = 100000.0 #2097152.0/2.0
	chi_sq_step = []
	
	for iter in range(1000):
		print('\n(Iteration, lambda)= ',[iter,lmd])
		pars_new,step,lhs = newton(f,pars_cur,delete_index,y_data,Ninv,dm,lmd)
		y_pred = get_spectrum(pars_new,n)
		chi_sq_new = chi_sq(y_data,y_pred,error_data)
		if chi_sq_new < chi_sq_curr:
			chi_sq_step.append(chi_sq_new)
			pars_cur = pars_new
			lhs_cur = lhs
			y_cur = get_spectrum(pars_cur,n)
			chi_sq_curr = chi_sq_new
			lmd = lmd**0.5
			if lmd < 1.1:
				lmd = 0.0
		else:
			print('Chi_sq- new:',chi_sq_new,' curr:',chi_sq_curr)
			if lmd == 0:	
				lmd += 2.0
			else:
				lmd = lmd*2	
		if True:
			print('\n\nFinal parameter values: \n',pars_cur)
			break
	return pars_cur, lhs_cur, chi_sq_step	

def chi_sq(data,y,err):
	# X,data are the given data, y is obtained from fit using 'pars'
	return np.sum((data-y)**2/err**2)

def f(params,n,dx):
	y=get_spectrum(params,n)
	derivs=np.zeros([n,len(params)])
	for i in range(len(params)):		
		par = copy.deepcopy(params)
		par[i] += dx[i]
		y2=get_spectrum(par,n)
		par[i] -= 2*dx[i]
		y1=get_spectrum(par,n)
		derivs[:,i]=(y2-y1)/(2*dx[i])
	return y,derivs

def run_mcmc2(pars,data,corr_mat,prior,chifun,file,nstep=5000):
	# Modified mcmc function to add prior for tau
	prior_chi =  ((pars[3]-prior[0])/prior[1])**2 # Prior chi-square
	file = open(file,'w')
	npar=len(pars)
	chain=np.zeros([nstep,npar])
	chivec=np.zeros(nstep)
	y_pred = get_spectrum(pars,len(data[0]))
	chi_cur=chifun(data[1],y_pred,data[2])+prior_chi # The prior is added to the result of chifun
	L = 0.7*np.linalg.cholesky(corr_mat)	#Scaled matrix to suppress step size and improve acceptance
	for i in range(nstep):
		dpar = L@np.random.randn(npar)
		pars_trial=pars+dpar
		print('\nStep ',i,'  Trial parameter: \n',pars_trial)
		if pars_trial[3]<0.0:
			continue
		y_new = get_spectrum(pars_trial,len(data[0]))		
		prior_chi =  ((pars_trial[3]-prior[0])/prior[1])**2		
		chi_trial=chifun(data[1],y_new,data[2])+prior_chi  # Trail chi-sq with prior included
		#we now have chi^2 at our current location
		#and chi^2 in our trial location. decide if we take the step
		accept_prob=np.exp(-0.5*(chi_trial-chi_cur))
		if np.random.rand(1)<accept_prob: #accept the step with appropriate probability
			pars=pars_trial
			chi_cur=chi_trial
		chain[i]=pars
		chivec[i]=chi_cur
		file.write(str(chivec[i])+' '+str(chain[i,0])+' '+str(chain[i,1])+' '+
			 str(chain[i,2])+' '+str(chain[i,3])+' '+str(chain[i,4])+' '+str(chain[i,5])+' '+'\n')
	file.close()
	return chain,chivec


###############################################################################

pars_ini=np.asarray([65.,0.02,0.1,0.05,2e-9,0.96])
wmap=np.loadtxt('WMAP_data.txt')
error_data = wmap[:,2]
y_data = wmap[:,1]

# Prep of initial conditions for parameters and other variables for Newton/LM Minimizer
#pars_cur=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
pars_cur = np.asarray([7.10138295e+01,2.22880134e-02,1.07300856e-01,5.06943875e-02,1.99554631e-09,9.73364402e-01])
#pars_cur=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
delete_index = []
pars_cur, lhs, chi = LM(pars_cur,y_data,error_data,delete_index,f)

# Using above results to seed mcmc
data=wmap.T
prior = [0.0544,0.0073]
file = 'chain_prior_final1.txt'
chain,chivec=run_mcmc2(pars_cur,data,np.linalg.inv(lhs),prior,chi_sq,file,nstep=5000)

chain  = []
chivec = []
file = open('chain_prior_final1.txt','r')
for t in file:
	t = np.asarray(t.strip().split(),dtype='float')
	chivec.append(t[0])
	chain.append(t[1:])
file.close()	
chain = np.asarray(chain)	
chivec = np.asarray(chivec)


file = 'chain_prior_final2.txt'
pars_guess=np.median(chain,axis=0)
for i in range(chain.shape[1]):
    chain[:,i]=chain[:,i]-pars_guess[i]
mycov=chain.T@chain/chain.shape[0]
chain2,chivec2=run_mcmc2(pars_guess,data,mycov,prior,chi_sq,file,nstep=20000)

chain2  = []
chivec2 = []
file = open('chain_prior_final2.txt','r')
for t in file:
	t = np.asarray(t.strip().split(),dtype='float')
	chivec2.append(t[0])
	chain2.append(t[1:])
file.close()	
chain2 = np.asarray(chain2)	
chivec2 = np.asarray(chivec2)

pars_cur=np.median(chain2,axis=0)
delt = (chain2).copy()
for i in range(delt.shape[1]):
    delt[:,i]=delt[:,i]-pars_cur[i]
mycov=delt.T@delt/delt.shape[0]
par_errs=np.abs(np.sqrt(np.diag(mycov)))
print('Current parameter values:')
for i in range(6):
    print('Parameter'+str(i)+' : ',pars_cur[i],'+/-',par_errs[i])

#-- IMportance sampling -------------------------------------
chain_temp  = []
chivec_temp = []
# Reading data of previous question
file = open('chain_data_final.txt','r')
for t in file:
	t = np.asarray(t.strip().split(),dtype='float')
	chivec_temp.append(t[0])
	chain_temp.append(t[1:])
file.close()	
chain_temp = np.asarray(chain_temp)	
chivec_temp = np.asarray(chivec_temp)

# Initiating variables 
chain_imp_scat = chain_temp.copy()
chain_imp_means=np.zeros(chain_temp.shape[1])
chain_imp_errs=np.zeros(chain_temp.shape[1])
# Calculating weights associated to the prior defined earlier
wtvec = np.exp(-0.5*((chain_temp[:,3]-prior[0])/prior[1])**2)

for i in range(chain_temp.shape[1]):
	chain_imp_means[i]=np.sum(wtvec*chain_temp[:,i])/np.sum(wtvec)	# Means from importance sampled data
	chain_imp_scat[:,i]=chain_imp_scat[:,i]-chain_imp_means[i]		# Subtracting out the means
	chain_imp_errs[i]=np.sqrt(np.sum(chain_imp_scat[:,i]**2*wtvec)/np.sum(wtvec))	# Calculating standard deviation

pars_cur=chain_imp_means
par_errs=chain_imp_errs
print('Current parameter values and errors (from importnace sampling):')
for i in range(6):
    print('Parameter'+str(i)+' : ',pars_cur[i],'+/-',par_errs[i])

'''
chain2  = []
chivec2 = []
file = open('chain_data2.txt','r')
for t in file:
	t = np.asarray(t.strip().split(),dtype='float')
	chivec2.append(t[0])
	chain2.append(t[1:])
chain2 = np.asarray(chain2)	
chivec2 = np.asarray(chivec2)
file.close()
'''