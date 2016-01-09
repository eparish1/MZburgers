from pylab import *
from padding import *
from myffts import *
import numpy as np
def initialConditions(x,kc,k):
  nx = size(x)
  u = zeros(size(x))
  np.random.seed(1)
  beta = fftshift( rand(size(x))*2*pi - pi)
  E = zeros(nx)
  Esum = 0
  for i in range(1,nx/2):
    if  ( k[i] >=1 and  k[i] <=5):
      E[i] = 5.**(-5./3.)
    else:
      if (k[i] < kc):
        E[i] = abs(k[i])**(-5./3.)
    #print(sqrt(2.*E)*sin(abs(k[i])*x + beta[i]) )
    u[:] = u[:] + sqrt(2.*E[i])*sin(abs(k[i])*x + beta[i]) 
    Esum += E[i]
  return u

def computeSGS(uhat,kc):
  N = (size(uhat)-1)*2
  G = ones(size(uhat))
  G[kc::] = 0
  uhat_filt = G*uhat
  uhat_pad = pad_r(uhat,1)
  uhat_filt_pad = pad_r(uhat_filt,1)
  ureal = myifft(uhat_pad)*sqrt(N)*3./2.
  u_filtreal = myifft(uhat_filt_pad)*sqrt(N)*3./2.
  c = unpad_r(myfft(ureal*ureal)/(3./2.*sqrt(N)),1)
  c_filt = unpad_r(myfft(u_filtreal*u_filtreal)/(3./2.*sqrt(N)),1)
  tauhat = 0.5*(G*c - c_filt)
  return tauhat
