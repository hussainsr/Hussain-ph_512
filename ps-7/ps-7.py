# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:58:33 2020

@author: hussa
"""
# ----------------------------------------------------------------------------
# Problem 2
# ----------------------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt
import time

plt.ion()

# Part A
# ----------------------------------------------------------------------------

n=256

V=np.zeros([n,n]) # Forming the grid

x=np.arange(n)
x[n//2:]=x[n//2:]-n
xx,yy=np.meshgrid(x,x) # Defining x, y values on the grid
xx[0,0]  = 1.0 
yy[0,0]  = 1.0 # Taking care of the origin by re-setting it

V = -np.log(np.sqrt(xx**2+yy**2))/(2*np.pi)

V[0,0] = V[1,0]*4-(V[2,0]+V[1,1]+V[1,-1]) # Finding V at origin

rho = V[0,0]-0.25*(V[1,0]+V[-1,0]+V[0,1]+V[0,-1]) # Calculating rho at origin

V = V/rho # Scaling potential to make rho = 1.0

rho = V[0,0]-0.25*(V[1,0]+V[-1,0]+V[0,1]+V[0,-1])

V = V - V[n//2,n//2] + 1. # Shifting potential to 1.0

print('(V[0,0], V[5,0]):  ',V[0,0],V[5,0])

plt.title('Potential from point charge at origin')
plt.imshow(V)


	
# Part B
# ----------------------------------------------------------------------------
# Code from laplace_fft.py (PDE folder)

def rho2pot(rho,kernelft):
    tmp=rho.copy()
    tmp=np.pad(tmp,(0,tmp.shape[0]))

    tmpft=np.fft.rfftn(tmp)
    tmp=np.fft.irfftn(tmpft*kernelft)
    if len(rho.shape)==2:
        tmp=tmp[:rho.shape[0],:rho.shape[1]]
        return tmp
    if len(rho.shape)==3:
        tmp=tmp[:rho.shape[0],:rho.shape[1],:rho.shape[2]]
        return tmp
    print("error in rho2pot - unexpected number of dimensions")
    assert(1==0)

def rho2pot_masked(rho,mask,kernelft,return_mat=False):
    rhomat=np.zeros(mask.shape)
    rhomat[mask]=rho
    potmat=rho2pot(rhomat,kernelft)
    if return_mat:
        return potmat
    else:
        return potmat[mask]


def cg(rhs,x0,mask,kernelft,niter,fun=rho2pot_masked,show_steps=False,step_pause=0.01):
    """cg(rhs,x0,mask,niter) - this runs a conjugate gradient solver to solve Ax=b where A
    is the Laplacian operator interpreted as a matrix, and b is the contribution from the 
    boundary conditions.  Incidentally, one could add charge into the region by adding it
    to b (the right-hand side or rhs variable)"""

    t1=time.time()
    Ax=fun(x0,mask,kernelft)
    r=rhs-Ax
    #print('sum here is ',np.sum(np.abs(r[mask])))
    p=r.copy()
    x=x0.copy()
    rsqr=np.sum(r*r)
    print('starting rsqr is ',rsqr)
    for k in range(niter):
        #Ap=ax_2d(p,mask)
        Ap=fun(p,mask,kernelft)
        alpha=np.sum(r*r)/np.sum(Ap*p)
        x=x+alpha*p
        if show_steps:            
            tmp=fun(x,mask,kernelft,True)
            plt.clf();
            plt.imshow(tmp,vmin=-2.1,vmax=2.1)
            plt.colorbar()
            plt.title('rsqr='+repr(rsqr)+' on iter '+repr(k+1))
            plt.savefig('laplace_iter_1024_'+repr(k+1)+'.png')
            plt.pause(step_pause)
        r=r-alpha*Ap
        rsqr_new=np.sum(r*r)
        beta=rsqr_new/rsqr
        p=r+beta*p
        rsqr=rsqr_new
        #print('rsqr on iter ',k,' is ',rsqr,np.sum(np.abs(r[mask])))
    t2=time.time()
    print('final rsqr is ',rsqr,' after ',t2-t1,' seconds')
    return x


		
n=1024
bc=np.zeros([n,n])
mask=np.zeros([n,n],dtype='bool')
mask[0,:]=True
mask[-1,:]=True
mask[:,0]=True
mask[:,-1]=True

# Defining the box
bc[4*n//9:5*n//9,4*n//9]=1.0
mask[n//4:3*n//4,(2*n//5)]=True

bc[4*n//9:5*n//9,5*n//9]=1.0
mask[4*n//9:5*n//9,5*n//9]=True

bc[4*n//9,4*n//9:5*n//9]=1.0
mask[4*n//9,4*n//9:5*n//9]=True

bc[5*n//9,4*n//9:5*n//9]=1.0
mask[5*n//9,4*n//9:5*n//9]=True

kernel=V
kernelft=np.fft.rfft2(kernel)

rhs=bc[mask]
x0=0*rhs

rho_out=cg(rhs,x0,mask,kernelft,40,show_steps=True,step_pause=0.25)
pot=rho2pot_masked(rho_out,mask,kernelft,True)


		
# Part C
# ----------------------------------------------------------------------------
