# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:14:46 2016

@author: ppycago
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.image as mpimg
plt.close()

#Gravitational lensing black hole lens 


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread('M101_hires_STScI-PRC2006-10a.jpg')     
sp = rgb2gray(img)  
plt.figure(1)  
plt.imshow(sp, cmap = plt.get_cmap('gray'))
plt.show()


#setup/ input values


#galaxy qualities black hole ?? 

def lens(sp):
    #units and constants
    #pc =(3.0857)*1e16 
    #c = 3e8 # m/s
    #vd = (1500)*1e3 # m/s  velocity dispersion
    #h = 0.7 # approx 
    #Rc=(70/h)*1e3*pc #metres, 70 original 
    e= 0 
    #Ds = (878/h)*1e6*pc   # metres distance to source 878 original
    #Dl = (637/h)*1e6*pc #metres distance to lens
    #Dls = (441/h)*1e6*pc #metres lens to source
    # note Dl + Dls doesnt equal  Ds
    #c = 3e8 # m/s
    
    #thetae= (4*np.pi*vd**2*Dls)/(c**2*Ds) # einstein radius
    rc = 0#Rc/(Dl*thetae) #small but not zero if zero sets up like a black hole 

    lp = np.zeros(sp.shape) # sets up the lens plane 
    xd = 1000
    yd = 1280

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
           lp[ix,iy] = sp[s1,s2]
    plt.figure(2)     
    plt.imshow(lp,extent = [np.min(r1), np.max(r1), np.min(r2), np.max(r2)], cmap='gray')

#
    
lens(sp)