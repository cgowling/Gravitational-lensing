# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:11:47 2016

@author: ppycago
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt 
from scipy import misc
import numpy as np
plt.close()
#Gravitational lensing
#setup/ input values


fnought = 5
x = np.linspace(-25,25)
y = np.linspace(-25,25)
r = [x,y]
a = 0.6
f =fnought* np.exp(-r/a)


vd = (1500)*1e3 # m/s  velocity dispersion
h = 0.7 # approx 
pc =(3.0857)*1e16
Rc = (70/h)*1e3*pc #metres, 70 original 
e = 0.1 #  ellipticity between 0 and 1, 0= sphere ? 
Ds = (878/h)*1e6*pc   # metres distance to source 878 original
Dl = (637/h)*1e6*pc #metres distance to lens
Dls = (441/h)*1e6*pc #metres lens to source
# note Dl + Dls doesnt equal  Ds
c = 3e8 # m/s
N = 100 # Mesh size
thetae= (4*np.pi*vd**2*Dls)/(c**2*Ds) # einstein radius
rc = Rc/(Dl*thetae)



sp = np.zeros([N,N]) # sets up the source plane
sp[N/2,N/2] = 1 # adds the sourece 
plt.imshow(sp)
lp = np.zeros([N,N]) # sets up the lens plane 

r1 = np.arange(0,N+1)/(N/2) -1 # centers on the origin and normalizes  the radii
r2 = np.arange(0,N+1)/(N/2) -1

for ix in range(N):
   for iy in range(N):
       
       s1norm = r1[ix] -((1-e)*r1[ix])/np.sqrt(rc**2 + (1-e)*(r1[ix]**2) + (1+e)*(r2[iy]**2))
       s2norm = r2[iy] -((1+e)*r2[iy])/np.sqrt(rc**2 + (1-e)*(r1[ix]**2) + (1+e)*(r2[iy]**2))
       if np.isnan(s1norm) :
           s1norm = 0 
           s2norm = 0 
       s1 = np.floor((s1norm +1)*N/2) # rounds down 
       s2 = np.floor((s2norm +1)*N/2)
       lp[ix,iy] = sp[s1,s2]
       
     
plt.figure(2)
plt.imshow(lp, extent = [np.min(r1), np.max(r1), np.min(r2), np.max(r2)])
    

# first test with an alligned source and spherical lens 
#%%
# things for the furure
#plt.figure(1)
#sp1 = misc.imread('NGC_4414_(NASA-med).jpg')

#plt.show()

#def build_checkerboard(w, h) :
#      re = np.r_[ w*[0,1] ]              # even-numbered rows
#      ro = np.r_[ w*[1,0] ]              # odd-numbered rows
#      return np.row_stack(h*(re, ro))
#sp = build_checkerboard(N, N) 