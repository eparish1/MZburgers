from pylab import *
import os
sys.path.append("../../src")
from myffts import *
import numpy as np
from utilities import *
turb_model =0
N = 2048
kc = 16
dx = 2*pi/N
x = linspace(0,2*pi-dx,N)
k = linspace(0,N/2,N/2+1)
#k = fftshift(linspace(-N/2,N/2-1,N))
u = initialConditions(x,kc,k)
uhat = myfft(u)/sqrt(N)
t = 0
dt = 1e-4 
et =2  
nu = 1e-2
live_plot = 0

execfile('../../src/burgersDriver.py')
