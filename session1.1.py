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

#Gravitational lensing

#plt.figure(1)
#sp = misc.imread('NGC_4414_(NASA-med).jpg')
#plt.imshow(sp)
#plt.show()



#def build_checkerboard(w, h) :
#      re = np.r_[ w*[0,1] ]              # even-numbered rows
#      ro = np.r_[ w*[1,0] ]              # odd-numbered rows
#      return np.row_stack(h*(re, ro))
#sp = build_checkerboard(N, N)  



N = 100
Nhalf = int(N/2)
sp = np.zeros([N,N])
sp[(N -4)/2:(N+4)/2,(N -4)/2:(N+4)/2] = 1 # 4 by 4 "black" bit

lp = np.zeros([N+1,N+1])
for ix in range(-Nhalf,Nhalf):
    for iy in range (-Nhalf,Nhalf):
        lp[ix,iy] = ix,iy
    




vd = 1500# km/s  velocity dispersion
hubble = 0.7 # approx 
Rc = 70#*(1/h) # kpc
e = 0 #  ellipticity between 0 and 1 0= sphere ? 
Ds = 878#*(1/h) # Mpc distance to source
Dl = 637#*(1/h)# Mpc distance to lens
Dls = 441#*(1/h)# Mpc lens to source
c = 3e8# m/s
# note Dl + Dls doesnt equal  Ds

#thetae= (4*np.pi*vd**2*Dls)/(c**2*Ds) # einstein radius
#rc = Rc/(Dl*thetae)
#
#s1 = r1 -((1-e)*r1)/sqrt(rc**2 + (1-e)*r1**2 + (1+e)*r2**2)
#
#s2 = r2 -((1+e)*r2)/sqrt(rc**2 + (1-e)*r1**2 + (1+e)*r2**2)


# first test with an alligned source and spherical lens 
