# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:17:12 2016

@author: ppycago
"""


from IPython import get_ipython
get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt 
from scipy import misc
import numpy as np
import matplotlib.widgets as widgets
plt.close()

#Gravitational lensing

#setup/ input values
#units and constants
pc =(3.0857)*1e16 
c = 3e8 # m/s
#source galaxy
N = 250# Mesh size
fnought = 10
Nhalf =int( N/2)
x = np.linspace(-Nhalf,Nhalf, N)
y = np.linspace(-Nhalf,Nhalf,N)
[X,Y] = np.meshgrid(x,y)
r = np.sqrt(X**2 +Y**2)
a = 3
sp =fnought*np.exp(-r/a)

#galaxy qualities

vd = (1500)*1e3 # m/s  velocity dispersion
h = 0.7 # approx 
e = 0 
Rc = (70/h)*1e3*pc #metres, 70 original 
#e =  sliderHandleElip.val #  ellipticity between 0 and 1, 0= sphere ? 
Ds = (878/h)*1e6*pc   # metres distance to source 878 original
Dl = (637/h)*1e6*pc #metres distance to lens
Dls = (441/h)*1e6*pc #metres lens to source
# note Dl + Dls doesnt equal  Ds

thetae= (4*np.pi*vd**2*Dls)/(c**2*Ds) # einstein radius
rc = 1e-5 #Rc/(Dl*thetae) #small but not zero if zero sets up like a black hole 

lp = np.zeros([N,N]) # sets up the lens plane 

r1 = np.arange(0,N+1)/(N/2) -1 # centers on the origin and normalizes  the radii
r2 = np.arange(0,N+1)/(N/2) -1

fig,(ax1) = plt.subplots(1)
ax1, plt.imshow(lp, extent = [np.min(r1), np.max(r1), np.min(r2), np.max(r2)],cmap='gray')
plt.subplots_adjust(left=0.1, bottom=0.25,) 
def lens(sp):
    vd = (1500)*1e3 # m/s  velocity dispersion
    h = 0.7 # approx 
    global Rc#(70/h)*1e3*pc #metres, 70 original 
    global e 
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
         
    ax1.imshow(lp, extent = [np.min(r1), np.max(r1), np.min(r2), np.max(r2)],cmap='gray')

    fig.canvas.draw_idle()
    
    ax1.set_title('Lensed image')
    ax1.set_xlabel('radius')
    ax1.set_ylabel('radius')
    
def sliderCallbackElip(val): 
    global e
    e = sliderHandleElip.val
    lens(sp)
ax2 = plt.axes([0.2,0.09,0.65,0.03])
sliderHandleElip = widgets.Slider(ax2,'Elipticity', 0,1, valinit=0)    
sliderHandleElip.on_changed(sliderCallbackElip)

def sliderCallbackRc(val):
    global Rc
    Rc = sliderHandleRc.val
    lens(sp)
ax3 = plt.axes([0.2,0.04,0.65,0.03])
sliderHandleRc = widgets.Slider(ax3,'Core radius', (1e-10/h)*1e3*pc,(100/h)*1e3*pc, valinit=(70/h)*1e3*pc)    
sliderHandleRc.on_changed(sliderCallbackRc)

lens(sp)






