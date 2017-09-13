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
global e
global rc
N = 100# Mesh size
Nhalf =int( N/2)
#units and constants
pc =(3.0857)*1e16 
c = 3e8 # m/s
h = 0.7
#Lensing galaxy properties 

vd = (1500)*1e3 # m/s  velocity dispersion
h = 0.7 # approx 
Rc=(70/h)*1e3*pc# core radius 0 for black hole 
e= 0.1
Ds = (1400/h)*1e6*pc   # metres distance to source 878 original
Dl = (800/h)*1e6*pc #metres distance to lens
Dls = (600/h)*1e6*pc #metres lens to source
 #note Dl + Dls doesnt equal  Ds


thetae= (4*np.pi*vd**2*Dls)/(c**2*Ds) # einstein radius
rc = Rc/(Dl*thetae) 

def lens(sp):
    global rc
    global e
    lp = np.zeros([N,N]) # sets up the lens plane 

    r1 = np.arange(0,N+1)/(N/2) -1 # centers on the origin and normalizes  the radii
    r2 = np.arange(0,N+1)/(N/2) -1
    
    for ix in range(N):
        for iy in range(N):
           
           s1norm = r1[ix]-((1-e)*r1[ix])/np.sqrt(rc**2 + (1-e)*(r1[ix]**2) + (1+e)*(r2[iy]**2))
           s2norm = r2[iy] -((1+e)*r2[iy])/np.sqrt(rc**2 + (1-e)*(r1[ix]**2) + (1+e)*(r2[iy]**2))
           if np.isnan(s1norm):
               s1norm = 0 
               s2norm = 0 
           s1 = np.floor((s1norm +1)*N/2) # rounds down 
           s2 = np.floor((s2norm +1)*N/2)
           s1 = min(max(0, s1), N-1)
           s2 =  min(max(0, s2), N-1)
           lp[ix,iy] = sp[s1,s2]
    return lp



#source galaxy


fnought = 10
a = 3
x = np.linspace(-Nhalf,Nhalf, N)
y = np.linspace(-Nhalf,Nhalf,N)
[X,Y] = np.meshgrid(x,y)
caustics = np.zeros((N,N))

for xp in range(-Nhalf,Nhalf):
    for yp in range(-Nhalf,Nhalf): 

        r = np.sqrt((X-xp)**2 +(Y-yp)**2)
        sp =fnought*np.exp(-r/a)
        lp = lens(sp)
        keypoints = feature.blob_dog(lp, threshold=.5, max_sigma=20)
        nblobs = len(keypoints)
        caustics[xp-Nhalf,yp -Nhalf]= nblobs


# detects zero outside certain radius could be blob threshoold and sigma stuff isnt detectuing them in blob counter mounaklly as well       
plt.imshow(caustics)





