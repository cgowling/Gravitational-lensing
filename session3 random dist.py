# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:20:41 2016

@author: ppycago
"""


from IPython import get_ipython
get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt 
from scipy import misc
import numpy as np
import matplotlib.widgets as widgets
plt.close()

#Gravitational lensing random dist of galaxies
    
def lens(sp):
    vd = (1500)*1e3 # m/s  velocity dispersion
    h = 0.7 # approx 
    Rc=(70/h)*1e3*pc #metres, 70 original 
    e = 0.5
    Ds = (878/h)*1e6*pc   # metres distance to source 878 original
    Dl = (637/h)*1e6*pc #metres distance to lens
    Dls = (441/h)*1e6*pc #metres lens to source
    # note Dl + Dls doesnt equal  Ds
    c = 3e8 # m/s
    
    thetae= (4*np.pi*vd**2*Dls)/(c**2*Ds) # einstein radius
    rc = Rc/(Dl*thetae) #small but not zero if zero sets up like a black hole 

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
    plt.imshow(lp, extent = [np.min(r1), np.max(r1), np.min(r2), np.max(r2)],cmap='gray')

    
    
    plt.title('Lensed image')
    plt.xlabel('radius')
    plt.ylabel('radius')
    
#setup/ input values
#units and constants
pc =(3.0857)*1e16 
c = 3e8 # m/s

#source galaxy
N = 250# Mesh size
Ngal = 20#int(input('Insert an integer number of galaxies you would like in the source plane 1<Ngal<50'))
sp = np.zeros((N,N))
fnought = np.random.uniform(50,100,Ngal)
xnought = np.random.uniform(-N/2,N/2,Ngal) 
ynought = np.random.uniform(-N/2,N/2,Ngal)
a = np.random.uniform(0.1,8,Ngal) # sacle lengths
Nhalf = N/2
x = np.linspace(-Nhalf,Nhalf, N)
y = np.linspace(-Nhalf,Nhalf,N)

for n in range (Ngal):
    
    xs = x-xnought[n]
    ys = y - ynought[n]
    [X,Y] = np.meshgrid(xs,ys)
    r = np.sqrt(X**2 +Y**2)
    sp += fnought[n]*np.exp(-r/a[n])
    plt.imshow(sp,cmap='gray')
    #plt.pause(0.001)
    

    
lens(sp)    
    
    
    
    
    
    