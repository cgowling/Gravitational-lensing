# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:33:24 2016

@author: ppycago
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt 
import numpy as np

plt.close()
from skimage import feature
from matplotlib import pyplot as plt
#setup/ input values
#units
pc =(3.0857)*1e16 
c = 3e8 # m/s
h = 0.7
def lens(sp):
    #units and constants
    pc =(3.0857)*1e16 
    c = 3e8 # m/s
    vd = (1500)*1e3 # m/s  velocity dispersion
    h = 0.7 # approx 
    Rc=(70/h)*1e3*pc# core radius 0 for black hole 
    e= 0.1
    Ds = (1400/h)*1e6*pc   # metres distance to source 878 original
    Dl = (800/h)*1e6*pc #metres distance to lens
    Dls = (600/h)*1e6*pc #metres lens to source
     #note Dl + Dls doesnt equal  Ds
    c = 3e8 # m/s
    
    thetae= (4*np.pi*vd**2*Dls)/(c**2*Ds) # einstein radius
    rc = Rc/(Dl*thetae) #small but not zero if zero sets up like a black hole 

    lp = np.zeros(sp.shape) # sets up the lens plane 
    xd = sp.shape[0]
    yd = sp.shape[1]

    r1 = np.arange(0,xd+1)/(xd/2) -1 # centers on the origin and normalizes  the radii
    r2 = np.arange(0,yd+1)/(yd/2) -1
    
    for ix in range(xd):
        for iy in range(yd):
           
           s1norm = r1[ix]-((1-e)*r1[ix])/np.sqrt(rc**2 + (1-e)*(r1[ix]**2) + (1+e)*(r2[iy]**2))
           s2norm = r2[iy] -((1+e)*r2[iy])/np.sqrt(rc**2 + (1-e)*(r1[ix]**2) + (1+e)*(r2[iy]**2))
           if np.isnan(s1norm):
               s1norm = 0 
               s2norm = 0 
           s1 = np.floor((s1norm +1)*xd/2) # rounds down 
           s2 = np.floor((s2norm +1)*yd/2)
           s1 = min(max(0, s1), xd-1)
           s2 =  min(max(0, s2), yd-1)
           lp[ix,iy] = sp[s1,s2]
    return lp



#source galaxy

N = 10# Mesh size
fnought = 10
Nhalf =int( N/2)
a = 3
x = (np.linspace(-Nhalf,Nhalf, N))
y = np.linspace(-Nhalf,Nhalf,N)
caustics = np.zeros((N,N))
[X,Y] = np.meshgrid(x,y)
for xp in range(-Nhalf,Nhalf):
    for yp in range(-Nhalf,Nhalf): 

        
        r = np.sqrt((X-xp)**2 +(Y-yp)**2)
        
        sp =fnought*np.exp(-r/a)
        lp = lens(sp)
        keypoints = feature.blob_dog(lp, threshold=.5, max_sigma=20)
        nblobs = len(keypoints)
        caustics[xp+Nhalf,yp +Nhalf]= nblobs


      
plt.imshow(caustics)





