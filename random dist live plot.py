# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:47:20 2016

@author: cgowl_000
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt 
import numpy as np
plt.close()

#plotting random dist purttttty
N = 250# Mesh size
Ngal = 25#int(input('Insert an integer number of galaxies you would like in the source plane 1<Ngal<50'))
sp = np.zeros((N,N))
fnought = np.random.uniform(70,100,Ngal)# actual values range ? units????
xnought = np.random.uniform(-N/2,N/2,Ngal) 
ynought = np.random.uniform(-N/2,N/2,Ngal)
a = np.random.uniform(0.1,8,Ngal) # sacle lengths again actual values ???? units
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
    plt.pause(0.001)
#