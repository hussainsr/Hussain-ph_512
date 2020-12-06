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

plt.ion()

# Part A
# ----------------------------------------------------------------------------

n=2048

V=np.zeros([n,n]) # Forming the grid

x=np.arange(n)
x[n//2:]=x[n//2:]-n
xx,yy=np.meshgrid(x,x) # Defining x, y values on the grid
xx[0,0]  = 1.0 
yy[0,0]  = 1.0 # Taking care of the origin by re-setting it

V = -np.log(np.sqrt(xx**2+yy**2))/(2*np.pi) # Potential in 2d 

V[0,0] = V[1,0]*4-(V[2,0]+V[1,1]+V[1,-1]) # Finding V at origin

rho = V[0,0]-0.25*(V[1,0]+V[-1,0]+V[0,1]+V[0,-1]) # Calculating rho at origin

V = V/rho # Scaling potential to make rho = 1.0

rho = V[0,0]-0.25*(V[1,0]+V[-1,0]+V[0,1]+V[0,-1])

V = V - V[0,0] + 1. # Shifting potential at origin to 1.0

print('(V[0,0], V[5,0]):  ',V[0,0],V[5,0])

plt.title('Potential from point charge at origin')
plt.imshow(V)
plt.close()


	
# Part B
# ----------------------------------------------------------------------------
# Code from laplace_fft.py (PDE folder)

def rho2pot(rho,green_fft):
	tmp=rho.copy()
	tmp=np.pad(tmp,(0,tmp.shape[0]),mode = 'constant')
	tmpft=np.fft.rfft2(tmp)
	tmp=np.fft.irfft2(tmpft*green_fft)
	tmp=tmp[:rho.shape[0],:rho.shape[1]]
	return tmp

def rho2pot_masked(rho,mask,green_fft,return_mat=False):
     rhomat=np.zeros(mask.shape)
     rhomat[mask]=rho
     potmat=rho2pot(rhomat,green_fft)
     if return_mat:
          return potmat
     else:
          return potmat[mask]


def cg(rhs,x0,mask,green_fft,fun=rho2pot_masked):

     Ax=fun(x0,mask,green_fft)
     r=rhs-Ax
     p=r.copy()
     x=x0.copy()
     rsqr=np.sum(r*r)
     print('starting rsqr is ',rsqr)
     for k in range(100):
          Ap=fun(p,mask,green_fft)
          alpha=np.sum(r*r)/np.sum(Ap*p)
          x=x+alpha*p
#          tmp=fun(x,mask,green_fft,True)
#          plt.clf();
#          plt.imshow(tmp,vmin=-2.1,vmax=2.1)
#          plt.colorbar()
#          plt.title('rsqr='+repr(rsqr)+' on iter '+repr(k+1))
#          plt.pause(0.25/10.0)
          r=r-alpha*Ap
          rsqr_new=np.sum(r*r)
          beta=rsqr_new/rsqr
          p=r+beta*p
          rsqr=rsqr_new
     return x

n=n//2 		# Choosing smaller size to match with potential after padding
bc=np.zeros([n,n])
mask=np.zeros([n,n],dtype='bool')
mask[0,:]=True
mask[-1,:]=True
mask[:,0]=True
mask[:,-1]=True

# Defining the box
bc[4*n//9:5*n//9,4*n//9]=1.0
mask[4*n//9:5*n//9,4*n//9]=True

bc[4*n//9:5*n//9,5*n//9]=1.0
mask[4*n//9:5*n//9,5*n//9]=True

bc[4*n//9,4*n//9:5*n//9]=1.0
mask[4*n//9,4*n//9:5*n//9]=True

bc[5*n//9,4*n//9:5*n//9]=1.0
mask[5*n//9,4*n//9:5*n//9]=True

green=V
green_fft=np.fft.rfft2(green)

B=bc[mask]
x0=0*B

rho_out=cg(B,x0,mask,green_fft)
pot=rho2pot_masked(rho_out,mask,green_fft,True)
plt.title('Potential (from Box at V=1)')
plt.imshow(pot)
plt.colorbar()

rhomat=np.zeros(mask.shape)
rhomat[mask]=rho_out

change_den = rhomat[4*n//9,4*n//9-2:5*n//9+2]
#change_den = change_den/np.linalg.norm(change_den)
plt.title('Charge density along one side (extra padded to include points just outside)')
plt.plot(change_den)
plt.show()
		
# Part C
# ----------------------------------------------------------------------------

scale = 32 # Scale down the grid to see fewer points

Ex = np.zeros([n//scale,n//scale])
Ey = np.zeros([n//scale,n//scale])
E = np.gradient(pot) # Calculating the gradient of potential
Ey_temp = -E[0]
Ex_temp = -E[1]

for i in range(n//scale):
	for j in range(n//scale):		
		Ex[i,j] = Ex_temp[i*scale,j*scale] 
		Ey[i,j] = Ey_temp[i*scale,j*scale] 

plt.figure(figsize=[8,8])
plt.title('Electric field from the calculated potential')
X = np.linspace(-n//(2*scale),n//(2*scale)-1,n//scale)
U,V = np.meshgrid(X,X)
plt.quiver(U,V,Ex,Ey,width = 0.003,scale = 0.25)
plt.show()