# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 23:41:44 2020

@author: hussa
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()

def get_V(n, softening = 3.0, self_shield = 2.0): # Similar procedure as assignment 7 but in 3d and with softening
	V=np.zeros((n,n,n)) # Forming the grid
	x=np.arange(n)
	x[n//2:]=x[n//2:]-n
	xx,yy,zz=np.meshgrid(x,x,x) # Defining x, y values on the grid
	xx[0,0,0]  = 1.0 
	yy[0,0,0]  = 1.0            # Taking care of the origin by re-setting it
	zz[0,0,0]  = 1.0 

	V = -1/np.sqrt(xx*xx+yy*yy+zz*zz) # Potential in 3d 

	r = np.linspace(-softening, softening, int(softening)*2+1) # Softening isnt supercritical here because self shielding takes care of the close range interaction
	for i in r:
		for j in r:
			for k in r:
				if np.sqrt(i**2+j**2+k**2)<self_shield:         # Self shielding
					V[int(i),int(j),int(k)] = -(1/self_shield)  # Flattening the potential around particle location to make self-force=0
					continue
				if np.sqrt(i**2+j**2+k**2)<softening: 		# Softening
					V[int(i),int(j),int(k)] = -(3.*softening**2-(i**2+j**2+k**2))/(2.*softening**3)
	return V
	
	

class Particle:
	def __init__(self, x, v, g_size, m=1):
		self.x = x 			 	 	 	# Initial positions
		self.v = v 			 	 	 	# Initial velocities
		self.m = m 			 	 	 	# Paticle masses
		self.n = x.shape[0] 	 		# No. of particles
		self.f = np.empty(x.shape)		# Forces
		self.nn = g_size 				# Size of the 3d grid
	def get_forces(self,x,periodic):
		mask = np.zeros((self.nn,self.nn,self.nn)) 
		mask[x[:,0],x[:,1],x[:,2]] = self.m     # Setting position of particles onto the grid 
		self.f=0                                # Initialize force
		if periodic:
			V = get_V(self.nn)
			V_fft = np.fft.rfftn(V)	
			V_new = np.fft.irfftn(np.fft.rfftn(mask)*V_fft)*(-1)
		else:
			V = get_V(self.nn*2)
			V_fft = np.fft.rfftn(V)	
			mask=np.pad(mask,(0,mask.shape[0]),mode = 'constant') # Zero Padding
			V_new = np.fft.irfftn(np.fft.rfftn(mask)*V_fft)*(-1)
			V_new=V_new[:self.nn,:self.nn]		                  # Trimming the edges
		fx,fy,fz = np.gradient(V_new)
		self.f = np.transpose([fx[x[:,0],x[:,1],x[:,2]],fy[x[:,0],x[:,1],x[:,2]],fz[x[:,0],x[:,1],x[:,2]]])
		return (V_new[x[:,0],x[:,1],x[:,2]]+V[0,0,0])*self.m  #The V[0,0,0] is to remove the potential from itself
	def update(self,dt,periodic):
		x_temp = self.x+0.5*self.v*dt       # Moving by half step
		xxx = x_temp.copy()
		xxx = np.asarray(xxx,dtype = 'int')
		if periodic:
			xxx = xxx%self.nn               # wrap particles going outside the grid around
		U = self.get_forces(xxx,periodic)   # Obtaining the force at half step location
		v_temp = self.v + 0.5*dt*self.f
		self.x = self.x + dt*v_temp         # Updating location
		if periodic:
			self.x = self.x%self.nn         # wrap particles going outside the grid around
		self.v = self.v + dt*self.f         # Updating velocity
		K = 0.5*self.m*np.sum(v_temp[:,0]**2+v_temp[:,1]**2+v_temp[:,2]**2) # Kinetic energy
		return K,np.sum(U)/2.               # 1/2. factor is to compensate for double counting of potential energy


# ---------------------------------------------------------------------------
# Part 1
# ---------------------------------------------------------------------------
print("\n\nPart 1")
itter = 300			# Number of iterations/steps
n = 1				# Number of particles
g_size = 64			# Grid size
self_shield = 2  	# Self shielding (default)      
dt = self_shield**(-3/2)*0.05 # Time step based on the maximum possible velocity given self shielding is used
K = np.zeros(itter)
U = np.zeros(itter)

x = g_size//2+np.zeros((n,3)) 	# Setting particle location to center of grid
v = np.random.rand(n,3)*0.0		# Setting initial velocity to zero

pp = Particle(x,v,g_size) # Generating object of class particle for given initial conditions
for j in range(itter):
	for i in range(5):		
		K[j],U[j] = pp.update(dt,False)	# Updating the particle position for fixed timestep and non-periodic B.C.	

	plt.plot(pp.x[:,0],pp.x[:,1],'o')   # Visualizing particle in 2d
	plt.axis([0,g_size,0,g_size])
	plt.pause(0.5)
	
plt.close()	

# ---------------------------------------------------------------------------
# Part 2
# ---------------------------------------------------------------------------
print("\n\nPart 2")
itter = 200					# Number of steps
n = 2						# Number of particles
g_size = 64					# Grid size
self_shield = 2				# Not very important since orbiting particles wont come close to each other
dt = 0.5 					# For the separation I chose, dt=1 is small enough to avoid issues and speeds up the sim

E = np.zeros(itter)
K = np.zeros(itter)
U = np.zeros(itter)

x = [[g_size/3,g_size/2,g_size/2],[2*g_size/3,g_size/2,g_size/2]] # Particle position
x = np.asarray(x)
v = [[0,1,0],[0,-1,0]]
v = np.asarray(v)

# For circular orbit, v = sqrt(Gm^2/(R(2m))) for identical masses, here i set G=m=1, hence v=sqrt(1/(2d))
alpha = np.sqrt(0.5/(x[1,0]-x[0,0]))		
v = v*alpha

pp = Particle(x,v,g_size)
for j in range(itter):
	for i in range(5):		
		K[j],U[j] = pp.update(dt,False)		
	E[j] = (K[j]-U[j])

	plt.plot(pp.x[:,0],pp.x[:,1],'o')
	plt.axis([0,g_size,0,g_size])
	plt.pause(0.1)
plt.close()

# ---------------------------------------------------------------------------
# Part 2.1
# ---------------------------------------------------------------------------

itter = 200
n = 2
g_size = 64
self_shield = 2				
dt = self_shield**(-1.5)*0.05
E = np.zeros(itter)
U = np.zeros(itter)
K = np.zeros(itter)

x = [[g_size/2-2,g_size/2,g_size/2],[g_size/2+2,g_size/2,g_size/2]]
x = np.asarray(x)
v = [[0,1,0],[0,-1,0]]
v = np.asarray(v)*0 # Setting velocity to zero

pp = Particle(x,v,g_size)
for j in range(itter):
	for i in range(10):		
		K[j],U[j] = pp.update(dt,False)		

	plt.plot(pp.x[:,0],pp.x[:,1],'o')
	plt.axis([g_size/2-3,g_size/2+3,g_size/2-3,g_size/2+3])
	plt.pause(0.1)
plt.close()
E = -U[1:] + 0.5*(K[:-1]+K[1:])

# ---------------------------------------------------------------------------
# Part 3 (non periodic)
# ---------------------------------------------------------------------------
print("\n\nPart 3")
itter = 200
n = 100000
g_size = 128
self_shield = 2				
dt = self_shield**(-1.5)*0.05

x = g_size//2+(2*np.random.rand(n,3)-1)*g_size/4.  # Shifting all points to a central location to prevent exit
x = np.asarray(x)
v=np.random.rand(n,3)*0
v = np.asarray(v)
E = np.zeros(itter)
K = np.zeros(itter)
U = np.zeros(itter)

pp = Particle(x,v,g_size) 

for j in range(itter):
	print(j)
	for i in range(5):		
		K[j],U[j] = pp.update(dt,False)		
	
	fig=plt.figure(figsize=(10,10))#Create 3D axes
	try: ax=fig.add_subplot(111,projection="3d")
	except : ax=Axes3D(fig) 
	ax.set_xlim3d(g_size//5,4*g_size//5)
	ax.set_ylim3d(g_size//5,4*g_size//5)
	ax.set_zlim3d(g_size//5,4*g_size//5)	
	ax.scatter(pp.x[:,0],pp.x[:,1], pp.x[:,2],color="royalblue",marker=".",s=.02)
	ax.set_xlabel("x-coordinate",fontsize=14)
	ax.set_ylabel("y-coordinate",fontsize=14)
	ax.set_zlabel("z-coordinate",fontsize=14)
	ax.set_title("3d n-body (iteration: "+str(j)+" )\n",fontsize=20)
	plt.savefig('images/part3-'+str(j)+'.png', dpi=50)	
	plt.close()	

E = -U[1:] + 0.5*(K[:-1]+K[1:])

# ---------------------------------------------------------------------------
# Part 3.1
# ---------------------------------------------------------------------------

itter = 200
n = 100000
g_size = 128
self_shield = 2				
dt = self_shield**(-1.5)*0.05

x = np.random.rand(n,3)*g_size
v = np.random.rand(n,3)*0
K = np.zeros(itter)
U = np.zeros(itter)

pp = Particle(x,v,g_size)

for j in range(itter):
	for i in range(5):		
		K[j],U[j] = pp.update(dt,True)		
	
	fig=plt.figure(figsize=(10,10))#Create 3D axes
	try: ax=fig.add_subplot(111,projection="3d")
	except : ax=Axes3D(fig) 
	ax.set_xlim3d(0,g_size)
	ax.set_ylim3d(0,g_size)
	ax.set_zlim3d(0,g_size)	
	ax.scatter(pp.x[:,0],pp.x[:,1], pp.x[:,2],color="royalblue",marker=".",s=.02)
	ax.set_xlabel("x-coordinate",fontsize=14)
	ax.set_ylabel("y-coordinate",fontsize=14)
	ax.set_zlabel("z-coordinate",fontsize=14)
	ax.set_title("3d n-body periodic (iteration: "+str(j)+" )\n",fontsize=20)
	plt.savefig('images/part3(1)-'+str(j)+'.png', dpi=50)	
	plt.close()	

E = -U[1:] + 0.5*(K[:-1]+K[1:])

# ----------------------------------------------------------------------------		
# Part 4
# ----------------------------------------------------------------------------		

g_size = 128
grid = np.random.rand(g_size,g_size,g_size) # Grid, White noise
grid_fft = np.fft.fftn(grid) 

x=np.fft.fftfreq(g_size)
kx,ky,kz=np.meshgrid(x,x,x)   # k definition
kx[0,0,0]=1.0
ky[0,0,0]=1.0
kz[0,0,0]=1.0 
Ak = (kx**2+ky**2+kz**2)**(-3/4) 	# Constructing the fourier realisation
Ak[0,0,0]=0.0						# Setting k = 0 to 0 (as opposed to 1/k^3 which tend to infinity)
Ak = Ak*grid_fft
m_den = np.real(np.fft.ifftn(Ak)) 	# Obtaining density in position/real space

plt.title('Density fluctuations for k^-3 case')
plt.imshow(m_den[:,:,64]) 			# Showing one slice

prob = (m_den-np.min(m_den))/(np.max(m_den)-np.min(m_den)) # Rescaling density to probablity

x = []
for i in range(g_size):
	for j in range(g_size):
		for k in range(g_size):
			if prob[i,j,k] > np.random.rand(): 	# Picking points based on probablity
				x.append([i,j,k]) 
x = np.asarray(x)				

g_size = 128
itter = 300
self_shield = 2				
dt = self_shield**(-1.5)*0.01
n = len(x)
v=np.random.rand(n,3)*0
v = np.asarray(v)

pp = Particle(x,v,g_size)

for j in range(itter):
	print(j)
	for i in range(5):		
		K,U = pp.update(dt,True)		# Periodic boundary conditions
	
	fig=plt.figure(figsize=(10,10))#Create 3D axes
	try: ax=fig.add_subplot(111,projection="3d")
	except : ax=Axes3D(fig) 
	ax.set_xlim3d(0,g_size)
	ax.set_ylim3d(0,g_size)
	ax.set_zlim3d(0,g_size)	
	ax.scatter(pp.x[:,0],pp.x[:,1], pp.x[:,2],color="royalblue",marker=".",s=.02)
	ax.set_xlabel("x-coordinate",fontsize=14)
	ax.set_ylabel("y-coordinate",fontsize=14)
	ax.set_zlabel("z-coordinate",fontsize=14)
	ax.set_title("Scale invariant power spectrum\n",fontsize=20)
	plt.savefig('images/part4-'+str(j)+'.png', dpi=100)	
	plt.close()	

 