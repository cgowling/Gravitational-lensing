# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 15:36:40 2016

@author: ppycago
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.image as mpimg
import matplotlib.widgets as widgets
plt.close()

# Lensing function 
def lens(sp,Rc,e):
    #constants
    pc =(3.0857)*1e16 
    c = 3e8 # m/s
    h = 0.7 # approx 
    #lens and source galxy distances 
    Ds = (1600/h)*1e6*pc   # metres distance to source 
    Dl = (800/h)*1e6*pc #metres distance to lens
    Dls = (500/h)*1e6*pc #metres lens to source
    #note Dl + Dls doesnt equal  Ds
    
    #lensing galaxy properties
    vd = (1500)*1e3 # m/s  velocity dispersion
    # einstein radius
    thetae= (4*np.pi*vd**2*Dls)/(c**2*Ds)

    #normalized core radius
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
        print('loading')

        
    plt.subplot(2,2,2)   
    plt.imshow(lp, cmap='gray',extent = [np.min(r1), np.max(r1), np.min(r2), np.max(r2)])
    plt.axis('image')
    plt.title('lensed Image')
    return lp,r1,r2
    
    
    
def sourceplane(sig):
    global N 
    global fnought
    global X
    global Y
    global a
    pc =(3.0857)*1e16 
    h = 0.7 # approx 
    if (sig ==1):#einstien ring
        r = np.sqrt((X)**2 +Y**2)
        
        sp =fnought*np.exp(-r/a)
        Rc =(70/h)*1e3*pc #metres, 70 original
        e = 0 
        plt.subplot(2,2,1)
        plt.imshow(sp,cmap='gray')
        plt.axis('image')
        ax5.cla()
        txt = '''
        Here we see an exponetialy decaying point
        source being lensed by a spherical galaxy
        (e=0) that is directly in front of it.
        This produces an Einstein ring with raidus 1
        Ds = 1600 Mpc/h distance to source 
        Dl = 800 Mpc/h  distance to lens
        Dls = 500 Mpc/h distance from lens to source
        vd = 1500 km/s  velocity dispersion'''
        ax5.text(0.01,0.5,txt)
        lens(sp,Rc,e)
        
        
    elif (sig ==2):#einstein cross
        r = np.sqrt((X-5)**2 +Y**2)
        sp =fnought*np.exp(-r/a)   
        Rc = (70/h)*1e3*pc #metres, 70 original 
        e =  0.2#  ellipticity between 0 and 1, 0= sphere ?  
        plt.subplot(2,2,1)
        plt.imshow(sp,cmap='gray')
        plt.axis('image')
        ax5.cla()
        txt = '''
        Here we see an exponetialy decaying
        point source slighty out of 
        allingnment with the lens being 
        lensed by an eliptical galaxy
        (e=0.2).
        This produces an Einstein cross.
        Ds = 1600 Mpc/h distance to source 
        Dl = 800 Mpc/h  distance to lens
        Dls = 500 Mpc/h distance from lens to source
        vd = 1500 km/s  velocity dispersion
        '''
        ax5.text(0.01,0.5,txt)
        
        lens(sp,Rc,e)
    elif (sig ==3): # random distribution
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
        Rc= (70/h)*1e3*pc #core radius
        e = 0.2  
        plt.subplot(2,2,1)
        plt.imshow(sp,cmap='gray')
        plt.axis('image')
        ax5.cla()
        txt = '''
        Here we see a random distribution of 50 galaxies 
        with varying scale lengths a and peak brightness. 
        This plane of galaxies has been lensed by a Galaxy 
        with a core radius of 70hkpc and elipticity of e = 0.2.
        Ds = 1600 Mpc/h distance to source 
        Dl = 800 Mpc/h  distance to lens
        Dls = 500 Mpc/h distance from lens to source
        vd = 1500 km/s  velocity dispersion
        
        '''
        ax5.text(0.01,0.5,txt)
        lens(sp,Rc,e)    
    
    elif (sig ==4):   # reding image in black hole lens
        print('starting read in')
        img = mpimg.imread('galaxy3.jpg')     
        sp = img[:,:,0]  
        Rc = 0 
        e = 0
        plt.subplot(2,2,1)
        plt.imshow(sp,cmap='gray')
        ax5.cla()
        txt = '''
        By placing applying the lensing function for the case of a 
        black hole to an image we can see the possible effects we may see.
        The core radius and elipticity are zero.
        Ds = 1600 Mpc/h distance to source 
        Dl = 800 Mpc/h  distance to lens
        Dls = 500 Mpc/h distance from lens to source
        vd = 1500 km/s
        
        '''
        ax5.text(0.01,0.5,txt)
        print('source image read in')
        lens(sp,Rc,e)
        
# lensing Galaxy properties and constants

#mesh size
N = 200
Nhalf =int( N/2)

# source galaxy properties
fnought = 100
x = np.linspace(-Nhalf,Nhalf, N)/2
y = np.linspace(-Nhalf,Nhalf,N)/2
[X,Y] = np.meshgrid(x,y)
a = 3# scale length 


fig = plt.figure()
ax3 = plt.axes([0.1,0.05,0.4,0.4])
R = widgets.RadioButtons(ax3,('Einstein Ring','Einstein Cross','Random distribution','Black hole lens'))
def selection(label):
    Rdict = {'Einstein Ring':1,'Einstein Cross':2,'Random distribution of source galaxies':3,'Black hole lens on a real image':4}
    sig = Rdict[label]
    sourceplane(sig)
    return sig
R.on_clicked(selection)


#Power button
def clickCallback(event):
    plt.close('all')
    
ax4 = plt.axes([0.01,0.7,0.1,0.1])# adds new axis to figure
buttonHandle = widgets.Button(ax4,'Power')
buttonHandle.on_clicked(clickCallback)# when the button is clicked on the callback function is called

ax5 = plt.axes([0.53,0.05,0.4,0.4])
ax5.axes.get_xaxis().set_visible(False)
ax5.axes.get_yaxis().set_visible(False)
txt = '''
Here we see an exponetialy decaying point
source being lensed by a spherical galaxy
(e=0) that is directly in front of it.
This produces an Einstein ring with raidus 1
Ds = 1600 Mpc/h distance to source 
Dl = 800 Mpc/h  distance to lens
Dls = 500 Mpc/h distance from lens to source
vd = 1500 km/s  velocity dispersion'''
ax5.text(0.01,0.35,txt)

fig.add_subplot(2,2,1)
r = np.sqrt(X**2 +Y**2)
sp0 =fnought*np.exp(-r/a)
plt.imshow(sp0,cmap='gray')
plt.title('Source Image')
fig.add_subplot(2,2,2)
lp = lens(sp0,0,0)
lp = lp[0]
plt.imshow(lp,cmap='gray')
plt.title('Lensed image')










