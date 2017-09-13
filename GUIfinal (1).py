# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 16:03:12 2016

@author: ppycago
Chloe Gowling
student id: 4233096
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.image as mpimg
import matplotlib.widgets as widgets
plt.close()

global sig
global fnought
global X
global Y
global a
global vd

# Lensing function 
def lens(sp,Rc,e,sig):
    
    #constants
    pc =(3.0857)*1e16 
    c = 3e8 # m/s
    h = 0.7 # approx 
    
    #lens and source galxy distances 
    
    Ds = (878/h)*1e6*pc   # metres distance to source 
    Dl = (637/h)*1e6*pc #metres distance to lens
    Dls = (441/h)*1e6*pc #metres lens to source
    #note Dl + Dls doesnt equal  Ds
    
    #lensing galaxy properties
    vd = (1500)*1e3 # m/s  velocity dispersion    
    thetae= (4*np.pi*vd**2*Dls)/(c**2*Ds)# einstein radius
    rc = Rc/(Dl*thetae)  #normalized core radius

    lp = np.zeros(sp.shape) # sets up the lens plane 
    xd = sp.shape[0]
    yd = sp.shape[1]

    r1 = np.arange(0,xd+1)/(xd/2) -1 # centers on the origin and normalizes the lens plane.
    r2 = np.arange(0,yd+1)/(yd/2) -1
    
    
    #for each position in the lens plane the following equations are used to 
    #calculate the source plane pixel that will appear at the current lensed plane position.
    #the following loop for each x value varies over all y values then move to next x value. 
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
        print((((ix)/xd)*100),'percent')
        ax3.cla()
        ax3.imshow(lp, cmap='gray')
        ax3.set_title('lensed image')
        fig.canvas.draw_idle()
        
        
    #teh following decides what axes to plot
    if 1<=sig<=3 :
        ax3.imshow(lp, cmap='gray',extent = [np.min(r1), np.max(r1), np.min(r2), np.max(r2)])
        ax3.set_ylabel('radius')
        ax3.set_xlabel('radius')
        
    elif sig ==4:
        ax3.imshow(lp,cmap= 'gray')
        ax3.axis('off')


    return lp
    
# This function, sets up different source planes depending on the value from the radio buttons.    
def sourceplane(sig):
    
    #mesh size
    N = 200
    Nhalf =int( N/2)
    
    # source galaxy properties
    fnought = 100
    global fnought
    global X
    global Y
    global a
    global vd
    sig = sig
    pc =(3.0857)*1e16 
    h = 0.7 # approx 
    if (sig ==1):# source plane for an einstien ring
        r = np.sqrt((X)**2 +Y**2)     
        sp =fnought*np.exp(-r/a)
        Rc =0
        e = 0 
        ax2.cla()
        ax2.imshow(sp,cmap='gray')
        ax2.set_title('Source image')        
        ax2.axis('off')
        ax5.cla()
        txt = '''
        Here we see an exponetially decaying
        point source being lensed by a 
        spherical galaxy (e=0) that is 
        directly in front of it.
        This produces an Einstein ring with 
        raidus 1.
        Ds = 1600 Mpc/h distance to source 
        Dl = 800 Mpc/h     "    to lens
        Dls = 500 Mpc/h    "   lens to source
        vd = 1500 km/s  velocity dispersion'''
        ax5.text(-0.03,0.2,txt,transform=ax5.transAxes)
        lens(sp,Rc,e,sig)
        fig.canvas.draw_idle()
        
        
    elif (sig ==2):# set up for an einstein cross
        r = np.sqrt((X-0.5)**2 +Y**2) # this will offset the source by 5 in the un-normalized scale.
        sp =fnought*np.exp(-r/a)   
        Rc = (70/h)*1e3*pc #metres
        e =  0.2 
        ax2.cla()
        ax2.imshow(sp,cmap='gray')
        ax2.axis('off')
        ax2.set_title('Source image')
        ax5.cla()
        txt = '''
        
        To produce the Einstein cross seen
        here, an exponetially decaying
        point source slighty out of 
        allingnment with an eliptical galaxy
        cluster lens with(e=0.2).
        Ds = 1600 Mpc/h distance to source 
        Dl = 800 Mpc/h     "     to lens
        Dls = 500 Mpc/h    " lens to source
        vd = 1500 km/s  velocity dispersion
        '''
        ax5.text(-0.03,0.2,txt,transform=ax5.transAxes)
        fig.canvas.draw_idle()
        lens(sp,Rc,e,sig)
        
    elif (sig ==3): # random distribution of source galaxies
        N = 200# Mesh size
        Ngal = 50
        sp = np.zeros((N,N))
        fnought1 = np.random.uniform(500,1000,Ngal)# assigns a random value of peak brightness
        xnought = np.random.uniform(-Nhalf,Nhalf,Ngal) # randomly gives x position of the galaxy
        ynought = np.random.uniform(-Nhalf,Nhalf,Ngal) # " for y
        a1 = np.random.uniform(0.1,8,Ngal) # scale lengths
        x = np.linspace(-Nhalf,Nhalf, N)
        y = np.linspace(-Nhalf,Nhalf,N)
        
        for n in range (Ngal):
    
            xs = x-xnought[n]
            ys = y - ynought[n]
            [X1,Y1] = np.meshgrid(xs,ys)
            r = np.sqrt(X1**2 +Y1**2)
            sp += fnought1[n]*np.exp(-r/a1[n]) # calculates a brigthness profile for each galaxy and adds it to the source plane
        ax2.cla()
        ax2.imshow(sp,cmap='gray')
        ax2.axis('image')
        ax2.set_title('Source image')
        Rc= (50/h)*1e3*pc #metres 
        e = 0.4
        ax5.cla()
        txt = '''
        Here we see a random distribution
        of 50 galaxies with varying scale
        lengths,a, and peak brightness,f0. 
        This plane of galaxies has been 
        lensed by a Galaxy cluster with a 
        core radius of 70hkpc and 
        elipticity of e = 0.2.
        Ds = 1600 Mpc/h distance to source 
        Dl = 800 Mpc/h    "     to lens
        Dls = 500 Mpc/h   "  lens to source
        vd = 1500 km/s  velocity dispersion
        '''
        ax5.text(0.01,0.01,txt,transform=ax5.transAxes)
        fig.canvas.draw_idle()
        lens(sp,Rc,e,sig)    
    
    elif (sig ==4):   # reding image in black hole lens
        print('starting read in')
        img = mpimg.imread('galaxy3.jpg')     
        sp = img[:,:,0]  
        Rc = 0 
        e = 0
        ax2.cla()       
        ax2.imshow(sp,cmap='gray')
        ax2.axis('off')
        ax2.set_title('Source image')
        ax5.cla()
        txt = '''
        We see here the effects of
        applying the lensing function 
        for the case of a black hole,
        e = 0 and Rc = 0,to an image of 
        a spiral galaxy.
        Ds = 1600 Mpc/h distance to source 
        Dl = 800 Mpc/h      "    to lens
        Dls = 500 Mpc/h     " lens to source
        vd = 1500 km/s velocity dispersion 
        
        '''
        ax5.text(0.01,0.1,txt,transform=ax5.transAxes)
        print('source image read in')
        ax2.axis('off')
        fig.canvas.draw_idle()
        lens(sp,Rc,e,sig)
        
# lensing Galaxy properties and constants

#mesh size
N = 200
Nhalf =int( N/2)

# source galaxy properties
fnought = 100
x = np.linspace(-Nhalf,Nhalf, N)
y = np.linspace(-Nhalf,Nhalf,N)
[X,Y] = np.meshgrid(x,y)
a = 3# scale length 

fig = plt.figure()

#Power button
def clickCallback(event):
    plt.close('all')
    
ax1 = plt.axes([0,0.75,0.1,0.1])
buttonHandle = widgets.Button(ax1,'Power')
buttonHandle.on_clicked(clickCallback)


ax2 = plt.subplot(2,2,1)
ax3 = plt.subplot(2,2,2)

# This sets up a radio button which allows the user to change what is lensed and the type of lensing.
ax4 = plt.axes([0.01,0.04,0.5,0.4])
R = widgets.RadioButtons(ax4,('Einstein Ring','Einstein Cross','Random distribution of galaxies','Black hole lens'))

def selection(label):
    global sig
    Rdict = {'Einstein Ring':1,'Einstein Cross':2,'Random distribution of galaxies':3,'Black hole lens':4}
    sig = Rdict[label]
    sourceplane(sig)
    return 
R.on_clicked(selection)

#An axis to display captions on.
ax5 = plt.axes([0.53,0.04,0.45,0.4])
ax5.axes.get_xaxis().set_visible(False)
ax5.axes.get_yaxis().set_visible(False)
txt = '''
Here we see an exponetially decaying
point source being lensed by a 
spherical galaxy (e=0) that is 
directly in front of it.
This produces an Einstein ring with 
raidus 1.
Ds = 1600 Mpc/h distance to source 
Dl = 800 Mpc/h     "    to lens
Dls = 500 Mpc/h    "   lens to source
vd = 1500 km/s  velocity dispersion'''
ax5.text(0.01,0.22,txt)

# Inital values so plots einstein ring first.
r = np.sqrt(X**2 +Y**2)
sp0 =fnought*np.exp(-r/a)
ax2.imshow(sp0,cmap='gray')
ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)
ax2.set_title('Source image')

lp = lens(sp0,0,0,1)
lp = lp[0]








