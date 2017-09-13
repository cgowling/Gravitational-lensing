# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:12:36 2016

@author: ppycago
"""


from IPython import get_ipython
get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.image as mpimg
from scipy import misc
import matplotlib.widgets as widgets
import time, sys
plt.close()

global sig
global N 
global fnought
global X
global Y
global a
# Lensing function 
def lens(sp,Rc,e):
    plt.subplot(2,2,2)
    plt.imshow(sp,cmap='gray')
    plt.axis('image')
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
        #update_progress()
    plt.subplot(2,2,4)   
    plt.imshow(lp, cmap='gray')
    plt.axis('image')
    plt.title('lensed Image')
    return lp,r1,r2
#extent = [np.min(r1), np.max(r1), np.min(r2), np.max(r2)],  
    
def sourceplane(sig):
    global N 
    global fnought
    global X
    global Y
    global a
    if (sig ==1):#einstien ring
        r = np.sqrt((X)**2 +Y**2)
        
        sp =fnought*np.exp(-r/a)
        Rc =(70/h)*1e3*pc #metres, 70 original
        e = 0 
        lens(sp,Rc,e)
        
    elif (sig ==2):#einstein cross
        r = np.sqrt((X-5)**2 +Y**2)
        sp =fnought*np.exp(-r/a)   
        Rc = (70/h)*1e3*pc #metres, 70 original 
        e =  0.2#  ellipticity between 0 and 1, 0= sphere ?    
        lens(sp,Rc,e)
    elif (sig ==3): #randm dist  ]Ngal = 100#int(input('Insert an integer number of galaxies you would like in the source plane 1<Ngal<50'))
        sp = np.zeros((N,N))
        Ngal =  50
        fnought = np.random.uniform(100,300,Ngal)# unitssss???
        xnought = np.random.uniform(-N/2,N/2,Ngal) 
        ynought = np.random.uniform(-N/2,N/2,Ngal)
        a = np.random.uniform(0.1,8,Ngal) # sacle lengths
    
        for n in range (Ngal):
            
            xs = x-xnought[n]
            ys = y - ynought[n]
            [X,Y] = np.meshgrid(xs,ys)
            r = np.sqrt(X**2 +Y**2)
            sp += fnought[n]*np.exp(-r/a[n])
        Rc= (70/h)*1e3*pc #metres, 70 original 
        e = 0.4        
        lens(sp,Rc,e)    
    
    elif (sig ==4):   # reding image in 
        img = mpimg.imread('galaxy3.jpg')     
        sp = img[:,:,0]  
        Rc = 0 
        e = 0
        lens(sp,Rc,e)


#def update_progress(progress):
#    barLength = 200 # Modify this to change the length of the progress bar
#    status = ""
#    if isinstance(progress, int):
#        progress = float(progress)
#    if not isinstance(progress, float):
#        progress = 0
#        status = "error: progress var must be float\r\n"
#    if progress < 0:
#        progress = 0
#        status = "Halt...\r\n"
#    if progress >= 1:
#        progress = 1
#        status = "Done...\r\n"
#    block = int(round(barLength*progress))
#    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
#    sys.stdout.write(text)
#    sys.stdout.flush()
    
       
    
  
# lensing Galaxy properties and constants

#constants
pc =(3.0857)*1e16 
c = 3e8 # m/s
h = 0.7 # approx 
#lens and source galxy distances 
Ds = (1200/h)*1e6*pc   # metres distance to source 878 original
Dl = (800/h)*1e6*pc #metres distance to lens
Dls = (500/h)*1e6*pc #metres lens to source
#note Dl + Dls doesnt equal  Ds
vd = (1500)*1e3 # m/s  velocity dispersion
N = 200
fnought = 100
Nhalf =int( N/2)
x = np.linspace(-Nhalf,Nhalf, N)/2
y = np.linspace(-Nhalf,Nhalf,N)/2
[X,Y] = np.meshgrid(x,y)
a = 3# scale length 

fig = plt.figure()
fig.add_subplot(2,2,2)
#r = np.sqrt((X)**2 +Y**2)
#sp =fnought*np.exp(-r/a)
plt.title('Source Image')
fig.add_subplot(2,2,4)
#imgplot =plt.imshow(sp,cmap='gray')
#lp = lens(sp,0,0)
#imgplot = plt.imshow(lp,cmap='gray')

plt.title('Lensed image')
#imgplot.set_clim(0.1,1)


ax3 = plt.axes([0.001,0.5,0.4,0.3])
R = widgets.RadioButtons(ax3,('Einstein Ring','Einstein Cross','Random distribution of source galaxies','Black hole lens on a real image'))
def selection(label):
    global sig
    Rdict = {'Einstein Ring':1,'Einstein Cross':2,'Random distribution of source galaxies':3,'Black hole lens on a real image':4}
    sig = Rdict[label]
    sourceplane(sig)
    return 
R.on_clicked(selection)




    
    
    