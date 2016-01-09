from pylab import *
def upwind(u,dx):
    Fxp = zeros(u.size)
    Fxm = zeros(u.size)
    Fplus = 0.5*(u + abs(u))/2.*u
    Fminus = 0.5*(u - abs(u))/2.*u
    Fxp[1:N] = 1./dx*(Fplus[1:N] - Fplus[0:N-1])
    Fxp[0] = 1./dx*(Fplus[0] - Fplus[-1])
    Fxm[0:N-1] = 1./dx*(Fminus[1:N] - Fminus[0:N-1])
    Fxm[-1] = 1./dx*(Fminus[0] - Fminus[-1])
    F = Fxp + Fxm
    return F

def viscousFlux(nu,u,dx):
    Fv = zeros(u.size)
    Fv[1:nx-1] = nu/dx**2*(u[2:nx] - 2.*u[1:nx-1] + u[0:nx-2])
    Fv[0] = nu/dx**2*(u[1] - 2.*u[0] + u[-1])
    Fv[-1] = nu/dx**2*(u[0] - 2.*u[-1] + u[-2])
    return Fv

