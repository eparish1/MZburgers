from pylab import *

## Three Halves Padding for Real FTs. k goes from [0:N/2+1]
#---------------------------------------------
def unpad_r(uhat_pad,arrange):
  N1 = int( size(uhat_pad)*2./3. + 1 )
  uhat = zeros((N1),dtype = 'complex')
  if (arrange == 1):
    uhat[0:N1-1] = uhat_pad[0:N1-1]
  return uhat

def pad_r(uhat,arrange):
  N1 = size(uhat)
  uhat_pad = zeros(int(N1 + (N1-1)/2 ),dtype = 'complex')
  if (arrange == 1):
    uhat_pad[0:N1-1] = uhat[0:N1-1]
  return uhat_pad
#--------------------------------------------

## Double padding for Real FTs.
#--------------------------------------------
def unpad_2xr(uhat_pad,arrange):
  N1 = int( size(uhat_pad-1)/2. + 1 )
  uhat = zeros((N1),dtype = 'complex')
  if (arrange == 1):
    uhat[0:N1-1] = uhat_pad[0:N1-1]
  return uhat

def pad_2xr(uhat,arrange):
  N1 = size(uhat)
  uhat_pad = zeros(int(2*N1 - 1 ),dtype = 'complex')
  if (arrange == 1):
    uhat_pad[0:N1-1] = uhat[0:N1-1]
  return uhat_pad

#--------------------------------------------

## Three Halves Pading for regular FTs. k goes from 0:N/2-1 then N/2:-1
def unpad(uhat_pad,arrange):
  N1 = int(size(uhat_pad)*2./3)
  uhat = zeros((N1),dtype = 'complex')
  if (arrange == 0):
   uhat[:] = uhat_pad[N1/4:N1/4+N1]
  if (arrange == 1):
    uhat[0:N1/2] = uhat_pad[0:N1/2]
    uhat[N1/2+1::] = uhat_pad[int(3./2*N1)-N1/2+1::]
  return uhat

def pad(uhat,arrange):
  N1 = size(uhat)
  uhat_pad = zeros(int(3./2*N1),dtype = 'complex')
  if (arrange == 0):
    uhat_pad[N1/4:N1/4+N1] = uhat[:]
  if (arrange == 1):
    uhat_pad[0:N1/2] = uhat[0:N1/2]
    uhat_pad[int(3./2*N1)-N1/2+1::] = uhat[N1/2+1::]
  return uhat_pad
##--------------------------------------------

## Double padding for regular FTs. 
#================================================
def unpad_2x(uhat_pad,arrange):
  N1 = size(uhat_pad)/2
  uhat = zeros((N1),dtype = 'complex')
  if (arrange == 0):
    uhat[:] = uhat_pad[N1/2:N1/2+N1]
  if (arrange == 1):
    uhat[0:N1/2] = uhat_pad[0:N1/2]
    uhat[N1/2+1::] = uhat_pad[2*N1-N1/2+1::]
  return uhat

def pad_2x(uhat,arrange):
  N1 = size(uhat)
  uhat_pad = zeros((2*N1),dtype = 'complex')
  if (arrange == 0):
    uhat_pad[N1/2:N1/2+N1] = uhat[:]
  if (arrange == 1):
    uhat_pad[0:N1/2] = uhat[0:N1/2]
    uhat_pad[2*N1-N1/2+1::] = uhat[N1/2+1::]
  return uhat_pad
#===============================================
