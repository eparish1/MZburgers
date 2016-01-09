from pylab import *
from padding import *
import numpy as np
from myffts import *
 
def RHS_DNS(main):
    N = (size(main.uhat) - 1)*2
    uhat_pad = pad_r(main.uhat,1)
    ureal = myifft(uhat_pad)*sqrt(N)*3./2.
    c = unpad_r(myfft(ureal*ureal)/(3./2.*sqrt(N)),1)
    main.RHS = -0.5*1j*main.k*c - main.nu*main.k**2*main.uhat
    main.tauhat = computeSGS(main.uhat,main.kc)
    main.u = myifft(main.uhat)*sqrt(N)
 

def splitModes(u_FUG,kc):
    u_G = zeros(size(u_FUG),dtype='complex')
    u_F = zeros(size(u_FUG),dtype='complex')
    u_G[:] = u_FUG[:]
    ## Splitting Modes. Note that the resolved modes don't include the oddball
    u_G[0:kc] = 0
    u_F[:] = u_FUG[:] - u_G[:]
    return u_F,u_G

def RHS_tmodel(main):
    N = (size(main.uhat) - 1)*2
    ## Compute basic RHS on a padded grid
    uhat_pad = pad_2xr(main.uhat,1)
    kf = linspace(0,size(uhat_pad)-1,size(uhat_pad))
    ureal = myifft(uhat_pad)*sqrt(N)*2.
    c_FUG = myfft(ureal*ureal)/(2.*sqrt(N))
    PLu_FUG = -0.5*1j*kf*c_FUG - main.nu*kf**2*uhat_pad
    ## Separate into modes in F and modes in G
    PLu_F,PLu_G = splitModes(PLu_FUG,main.kc)
    ## Now get pseudo-spectral conv(u,PLu_G)
    PLu_G_real = myifft(PLu_G)*sqrt(N)*2
    uhat_x_PLu_G = unpad_2xr(myfft(ureal*PLu_G_real)/(2.*sqrt(N)),1)
    ## t-model w/ 0.35 scaling
    PLQLu_F = 0.35* 2.*-1j*main.k/2.*uhat_x_PLu_G
    main.RHS = unpad_2xr(PLu_F,1) + main.t*PLQLu_F
    main.tauhat = main.t*PLQLu_F/(-1j*main.k + 1e-30)
    main.u[:] = myifft(main.uhat)*sqrt(N)

def RHS_FM1(main):
    N = (size(main.uhat) - 1)*2
    ## Compute basic RHS on a padded grid
    uhat_pad = pad_2xr(main.uhat,1)
    kf = linspace(0,size(uhat_pad)-1,size(uhat_pad))
    ureal = myifft(uhat_pad)*sqrt(N)*2.
    c_FUG = myfft(ureal*ureal)/(2.*sqrt(N))
    PLu_FUG = -0.5*1j*kf*c_FUG - main.nu*kf**2*uhat_pad
    ## Separate into modes in F and modes in G
    PLu_F,PLu_G = splitModes(PLu_FUG,main.kc)

    ## Now get pseudo-spectral conv(u,PLu_G)
    PLu_G_real = myifft(PLu_G)*sqrt(N)*2
    uhat_x_PLu_G = myfft(ureal*PLu_G_real)/(2.*sqrt(N))
    ## Now get PLQLu  
    PLQLu_FUG = 2.*-1j*kf/2.*uhat_x_PLu_G - kf**2*main.nu*PLu_G
 
    main.RHS[0::2] = unpad_2xr(PLu_F,1) + main.w0hat[:]
    main.RHS[1::2] = -2./main.dt0*main.w0hat[:] + 2.*unpad_2xr(PLQLu_FUG,1)
    main.tauhat[:] = main.w0hat[:]/(-1j*main.k + 1e-30)
    main.u[:] = myifft(main.uhat)*sqrt(N)


def RHS_FM2(main):
    N = (size(main.uhat) - 1)*2
    ## Compute basic RHS on a padded grid
    uhat_pad = pad_2xr(main.uhat,1)
    kf = linspace(0,size(uhat_pad)-1,size(uhat_pad))
    ureal = myifft(uhat_pad)*sqrt(N)*2.
    c_FUG = myfft(ureal*ureal)/(2.*sqrt(N))
    PLu_FUG = -0.5*1j*kf*c_FUG - main.nu*kf**2*uhat_pad
    ## Separate into modes in F and modes in G
    PLu_F,PLu_G = splitModes(PLu_FUG,main.kc)

    ## Now get pseudo-spectral conv(u,PLu_G)
    PLu_G_real = myifft(PLu_G)*sqrt(N)*2
    uhat_x_PLu_G = myfft(ureal*PLu_G_real)/(2.*sqrt(N))
    ## Now get PLQLu and split the modes 
    PLQLu_FUG = 2.*-1j*kf/2.*uhat_x_PLu_G - kf**2*main.nu*PLu_G
    PLQLu_F,PLQLu_G = splitModes(PLQLu_FUG,main.kc)

    ## Now get PLPLu and split the modes
    PLu_F_real = myifft(PLu_F)*sqrt(N)*2
    uhat_x_PLu_F = myfft(ureal*PLu_F_real)/(2.*sqrt(N))
    PLPLu_FUG = 2.*-1j*kf/2.*uhat_x_PLu_F - kf**2.*main.nu*PLu_F
    ## Now get PLQLQLu
    PLQLu_G_real = myifft(PLQLu_G)*sqrt(N)*2.
    uhat_x_PLQLu_G = myfft(ureal*PLQLu_G_real)/(2.*sqrt(N))
    PLu_G_x_PLu_F =  myfft(PLu_G_real*PLu_F_real)/(2.*sqrt(N))
    PLu_G_x_PLu_G =  myfft(PLu_G_real*PLu_G_real)/(2.*sqrt(N))
    PLQLQLu_FUG = 2.*-1j*kf/2.*uhat_x_PLQLu_G + 2.*-1j*kf/2.*PLu_G_x_PLu_F + \
              2.*-1j*kf/2.*PLu_G_x_PLu_G - main.nu*kf**2.*PLQLu_G;
    main.RHS[0::3] = unpad_2xr(PLu_F,1) + main.w0hat[:]
    main.RHS[1::3] = -2./main.dt0*main.w0hat[:] + 2.*unpad_2xr(PLQLu_F,1) + main.w1hat[:]
    main.RHS[2::3] = -2./main.dt1*main.w1hat[:] + 2.*unpad_2xr(PLQLQLu_FUG,1)
    main.tauhat[:] = main.w0hat[:]/(-1j*main.k + 1e-30)
    main.u[:] = myifft(main.uhat)*sqrt(N)

def RHS_FM3(main):
    N = (size(main.uhat) - 1)*2
    ## Compute basic RHS on a padded grid
    uhat_pad = pad_2xr(main.uhat,1)
    kf = linspace(0,size(uhat_pad)-1,size(uhat_pad))
    ureal = myifft(uhat_pad)*sqrt(N)*2.
    c_FUG = myfft(ureal*ureal)/(2.*sqrt(N))
    PLu_FUG = -0.5*1j*kf*c_FUG - main.nu*kf**2*uhat_pad
    ## Separate into modes in F and modes in G
    PLu_F,PLu_G = splitModes(PLu_FUG,main.kc)

    ## Now get pseudo-spectral conv(u,PLu_G)
    PLu_G_real = myifft(PLu_G)*sqrt(N)*2
    uhat_x_PLu_G = myfft(ureal*PLu_G_real)/(2.*sqrt(N))
    ## Now get PLQLu and split the modes 
    PLQLu_FUG = 2.*-1j*kf/2.*uhat_x_PLu_G - kf**2*main.nu*PLu_G
    PLQLu_F,PLQLu_G = splitModes(PLQLu_FUG,main.kc)

    ## Now get PLPLu and split the modes
    PLu_F_real = myifft(PLu_F)*sqrt(N)*2
    uhat_x_PLu_F = myfft(ureal*PLu_F_real)/(2.*sqrt(N))
    PLPLu_FUG = 2.*-1j*kf/2*uhat_x_PLu_F - kf**2.*main.nu*PLu_F
    PLPLu_F,PLPLu_G = splitModes(PLPLu_FUG,main.kc)

    ## Now get PLQLQLu
    PLQLu_G_real = myifft(PLQLu_G)*sqrt(N)*2.
    uhat_x_PLQLu_G = myfft(ureal*PLQLu_G_real)/(2.*sqrt(N))
    PLu_G_x_PLu_F =  myfft(PLu_G_real*PLu_F_real)/(2.*sqrt(N))
    PLu_G_x_PLu_G =  myfft(PLu_G_real*PLu_G_real)/(2.*sqrt(N))
    PLQLQLu_FUG = 2.*-1j*kf/2.*uhat_x_PLQLu_G + 2.*-1j*kf/2.*PLu_G_x_PLu_F + \
              2.*-1j*kf/2.*PLu_G_x_PLu_G - main.nu*kf**2.*PLQLu_G;
    PLQLQLu_F,PLQLQLu_G = splitModes(PLQLQLu_FUG,main.kc)

    PLQLQLu_G_real = myifft(PLQLQLu_G)*2.*sqrt(N)
    PLQLu_F_real = myifft(PLQLu_F)*2.*sqrt(N)
    PLPLu_F_real = myifft(PLPLu_F)*2.*sqrt(N)
    PLPLu_G_real = myifft(PLPLu_G)*2.*sqrt(N)
    uhat_x_PLQLQLu_G = myfft(ureal*PLQLQLu_G_real)/(2.*sqrt(N))
    PLu_F_x_PLQLu_G  = myfft(PLu_F_real*PLQLu_G_real)/(2.*sqrt(N))
    PLQLu_F_x_PLu_G  = myfft(PLQLu_F_real*PLu_G_real)/(2.*sqrt(N))
    PLPLu_F_x_PLu_G  = myfft(PLPLu_F_real*PLu_G_real)/(2.*sqrt(N))
    PLQLu_G_x_PLu_G  = myfft(PLQLu_G_real*PLu_G_real)/(2.*sqrt(N))
    PLPLu_G_x_PLu_G  = myfft(PLPLu_G_real*PLu_G_real)/(2.*sqrt(N))
    PLQLQLQLu_FUG = 2.*-1j*kf/2.*uhat_x_PLQLQLu_G + 4.*-1j*kf/2.*PLu_F_x_PLQLu_G + \
                4.*-1j*kf/2.*PLQLu_F_x_PLu_G + 2.*-1j*kf/2.*PLPLu_F_x_PLu_G + \
                6.*-1j*kf/2.*PLQLu_G_x_PLu_G + 2.*-1j*kf/2.*PLPLu_G_x_PLu_G - \
                main.nu*kf**2.*PLQLQLu_G;


    main.RHS[0::4] = unpad_2xr(PLu_F,1) + main.w0hat[:]
    main.RHS[1::4] = -2./main.dt0*main.w0hat[:] + 2.*unpad_2xr(PLQLu_F,1) + main.w1hat[:]
    main.RHS[2::4] = -2./main.dt1*main.w1hat[:] + 2.*unpad_2xr(PLQLQLu_FUG,1) + main.w2hat[:]
    main.RHS[3::4] = -2./main.dt2*main.w2hat[:] + 2.*unpad_2xr(PLQLQLQLu_FUG,1)

    main.tauhat[:] = main.w0hat[:]/(-1j*main.k + 1e-30)
    main.u[:] = myifft(main.uhat)*sqrt(N)
