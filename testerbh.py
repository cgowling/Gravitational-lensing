# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 22:06:27 2016

@author: cgowl_000
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt 

import numpy as np

plt.close()

N = 250# Mesh size
fnought = 10
Nhalf =int( N/2)
x = np.linspace(-Nhalf,Nhalf, N)
y = np.linspace(-Nhalf,Nhalf,N)
[X,Y] = np.meshgrid(x,y)
r = np.sqrt(X**2 +Y**2)
a = 3
sp =fnought*np.exp(-r/a)

def lens(sp):
    rc = 0 
    e = 0 
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
           lp[ix,iy] = sp[s1,s2]
    plt.figure(2)
    plt.imshow(lp,extent = [np.min(r1), np.max(r1), np.min(r2), np.max(r2)],cmap='gray')
lens(sp)