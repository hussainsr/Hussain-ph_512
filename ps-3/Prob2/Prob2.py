# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 20:22:17 2020

@author: hussa
"""
import numpy as np
import matplotlib.pyplot as plt
import camb
import matplotlib
matplotlib.rcParams.update({'font.size': 18})


def get_spectrum(pars,lmax=2000):
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
    return tt

# Function to calculate chi-sq
def chi_sq(data,y,err):
	# Comparing data with y from fitted parameters
	return np.sum((data-y)**2/err**2)

# Initializing paramters and obtaining corresponding power spectrum 
pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
wmap=np.loadtxt('WMAP_data.txt')
cmb=get_spectrum(pars)

plt.figure(figsize=(8,6))      
plt.title('Power spectrum')
plt.xlabel('Multipole')
plt.grid(alpha=0.75)
plt.plot(wmap[:,0],wmap[:,1],'.')
plt.plot(cmb)
plt.show()

# Calculating chi sq for the above parameters
chisq_fit = chi_sq(wmap[:,1],cmb[2:len(wmap)+2],wmap[:,2])
print('Lo and behold: ', chisq_fit)