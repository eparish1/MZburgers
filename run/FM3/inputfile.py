from pylab import *
import os
sys.path.append("../../src")
from myffts import *
import numpy as np
from utilities import *
turb_model = 4
N = 32
kc = 16
dx = 2*pi/N
x = linspace(0,2*pi-dx,N)
k = linspace(0,N/2,N/2+1)
DNSdata = load('DNSstats.npz')
n_DNS = (size(DNSdata['k']) - 1)*2
uhat = DNSdata['uhat'][0:17,0]*sqrt(N)/sqrt(n_DNS)
u = myifft(uhat)*sqrt(N)
#k = fftshift(linspace(-N/2,N/2-1,N))
#u = initialConditions(x,kc,k)
#uhat = myfft(u)/sqrt(N)
t = 0
dt = 1e-3 
et =2  
nu = 1e-2
live_plot = 1

execfile('../../src/burgersDriver.py')
