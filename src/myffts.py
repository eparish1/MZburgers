import numpy as np
def myfft(u):
  return np.fft.rfft(u)

def myifft(uhat):
  return np.fft.irfft(uhat)
