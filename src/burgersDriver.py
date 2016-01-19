from pylab import *
from padding import *
import numpy as np
from myffts import *
from RHSfunctions import *
close("all")
if not os.path.exists('Solution'):
   os.makedirs('Solution')

def advanceQ_RK4(main):
  Q0 = zeros(size(main.Q),dtype='complex')
  Q0[:] = main.Q[:]
  rk4const = array([1./4,1./3,1./2,1.])
  for i in range(0,4):
    main.computeRHS(main)
    main.Q = Q0 + main.dt*rk4const[i]*main.RHS
    main.Q2U()

## Setup Solution
class variables:
  def __init__(self,turb_model,N,k,uhat,u,t,kc,dt,nu):
    self.t = t
    self.kc = kc
    self.dt = dt
    self.nu = nu
    self.tauhat = zeros((N/2+1),dtype='complex')
    self.u = zeros(N)
    self.u[:] = u[:]
    self.uhat = zeros((N/2+1),dtype='complex')
    self.uhat[:] = uhat[:]
    self.turb_model = turb_model
    self.k = k
    self.Esave = 0.5*sum(self.uhat*conj(self.uhat))
    self.Dsave = 0
    self.Esave_resolved = 0.5*sum(self.uhat[0:self.kc]*conj(self.uhat[0:self.kc]))
    self.Dsave_resolved = 0

    #=======================================
    ## DNS Setup
    #=======================================
    if (turb_model == 0):
      self.Q = zeros((N/2+1),dtype='complex')
      self.Q[:] = uhat[:]
      def U2Q():
        self.Q[:] = self.uhat[:]
      def Q2U():
        self.uhat[:] = self.Q[:]
      self.U2Q = U2Q
      self.Q2U = Q2U
      self.computeRHS = RHS_DNS 

    #=======================================
    ## t-model setup
    #=======================================
    if (turb_model == 1):
      self.Q = zeros((N/2+1),dtype='complex')
      self.Q[:] = uhat[:]
      def U2Q():
        self.Q[:] = self.uhat[:]
      def Q2U():
        self.uhat[:] = self.Q[:]
      self.U2Q = U2Q
      self.Q2U = Q2U
      self.computeRHS = RHS_tmodel 

    #======================================
    ## First Order Finite Memory Setup
    #=======================================
    if (turb_model == 2):
      self.dt0 = 0.05
      self.Q = zeros(2*(N/2+1),dtype='complex')
      self.w0hat = zeros(size(self.uhat),dtype='complex')
      def U2Q():
        self.Q[0::2] = self.uhat[:]
        self.Q[1::2] = self.w0hat[:]
      def Q2U():
        self.uhat[:] = self.Q[0::2]
        self.w0hat[:] = self.Q[1::2]
      self.U2Q = U2Q
      self.Q2U = Q2U
      self.computeRHS = RHS_FM1 

    #======================================
    ## Second Order Finite Memory Setup
    #=======================================
    if (turb_model == 3):
      self.dt0 = 0.13
      self.dt1 = 0.07
      self.Q = zeros(3*(N/2+1),dtype='complex')
      self.w0hat = zeros(size(self.uhat),dtype='complex')
      self.w1hat = zeros(size(self.uhat),dtype='complex')
      def U2Q():
        self.Q[0::3] = self.uhat[:]
        self.Q[1::3] = self.w0hat[:]
        self.Q[2::3] = self.w1hat[:]
      def Q2U():
        self.uhat[:]  = self.Q[0::3]
        self.w0hat[:] = self.Q[1::3]
        self.w1hat[:] = self.Q[2::3]

      ## Make functions objects
      self.U2Q = U2Q
      self.Q2U = Q2U
      self.computeRHS = RHS_FM2 

    #======================================
    ## Third Order Finite Memory Setup
    #=======================================
    if (turb_model == 4):
      self.dt0 = 0.13
      self.dt1 = 0.07
      self.dt2 = 0.07
      self.Q = zeros(4*(N/2+1),dtype='complex')
      self.w0hat = zeros(size(self.uhat),dtype='complex')
      self.w1hat = zeros(size(self.uhat),dtype='complex')
      self.w2hat = zeros(size(self.uhat),dtype='complex')

      def U2Q():
        self.Q[0::4] = self.uhat[:]
        self.Q[1::4] = self.w0hat[:]
        self.Q[2::4] = self.w1hat[:]
        self.Q[3::4] = self.w2hat[:]
      def Q2U():
        self.uhat[:]  = self.Q[0::4]
        self.w0hat[:] = self.Q[1::4]
        self.w1hat[:] = self.Q[2::4]
        self.w2hat[:] = self.Q[3::4]


      self.w0Esave = sum(self.w0hat*conj(self.w0hat))
      self.w1Esave = sum(self.w1hat*conj(self.w1hat))
      self.w2Esave = sum(self.w2hat*conj(self.w2hat))
      ## Make functions objects
      self.U2Q = U2Q
      self.Q2U = Q2U
      self.computeRHS = RHS_FM3 


    if (turb_model == 5):
      self.dt0 = 0.13
      self.dt1 = 0.07
      self.dt2 = 0.07
      self.Q = zeros(4*(N/2+1),dtype='complex')
      self.w0hat = zeros(size(self.uhat),dtype='complex')
      self.w1hat = zeros(size(self.uhat),dtype='complex')
      self.w2hat = zeros(size(self.uhat),dtype='complex')
      def U2Q():
        self.Q[0::4] = self.uhat[:]
        self.Q[1::4] = self.w0hat[:]
        self.Q[2::4] = self.w1hat[:]
        self.Q[3::4] = self.w2hat[:]
      def Q2U():
        self.uhat[:]  = self.Q[0::4]
        self.w0hat[:] = self.Q[1::4]
        self.w1hat[:] = self.Q[2::4]
        self.w2hat[:] = self.Q[3::4]
      self.w0Esave = sum(self.w0hat*conj(self.w0hat))
      self.w1Esave = sum(self.w1hat*conj(self.w1hat))
      self.w2Esave = sum(self.w2hat*conj(self.w2hat))
      ## Make functions objects
      self.U2Q = U2Q
      self.Q2U = Q2U
      self.computeRHS = RHS_CM3 

    self.RHS = zeros(size(self.Q),dtype='complex')

uhat[-1] = 0
main = variables(turb_model,N,k,uhat,u,t,kc,dt,nu) 



ion()
itera = 0
uhatsave = zeros( (size(main.uhat),1),dtype = 'complex')
uhatsave[:,0] = main.uhat
tauhatsave = zeros( (size(main.uhat),1),dtype = 'complex')
tauhatsave[:,0] = computeSGS(main.uhat,main.kc)

usave = zeros( (size(main.u),1),dtype = 'complex')
usave[:,0] = main.u[:]
w0Esave = zeros(1)

tsave = zeros(0)
tsave = append(tsave,main.t)
tsave_full = zeros(0)
tsave_full = append(tsave_full,main.t)

while (main.t <= et):
  main.U2Q()
  advanceQ_RK4(main)
  main.t += dt
  print(main.t)
  tsave_full = append(tsave_full,main.t)

  #main.w0Esave = append(main.w0Esave,sum(main.w0hat*conj(main.w0hat)))
  #main.w1Esave = append(main.w1Esave,sum(main.w1hat*conj(main.w1hat)))
  #main.w2Esave = append(main.w2Esave,sum(main.w2hat*conj(main.w2hat)))

  main.Esave = append(main.Esave,0.5*sum(main.uhat*conj(main.uhat)) )
  main.Dsave = append(main.Dsave,1./dt*(main.Esave[-1] - main.Esave[-2]))
  main.Esave_resolved = append(main.Esave_resolved,0.5*sum(main.uhat[0:main.kc]*conj(main.uhat[0:main.kc])) )
  main.Dsave_resolved = append(main.Dsave_resolved,1./dt*(main.Esave_resolved[-1] - main.Esave_resolved[-2]))


  if (itera%10 == 0):

    tmp = zeros((size(main.uhat),1),dtype = 'complex')
    tmp[:,0] = main.uhat[:]
    uhatsave = append(uhatsave,tmp,1)
    tmp[:,0] = main.tauhat[:]
    tauhatsave =  append(tauhatsave,tmp,1)
    tmp = zeros((size(main.u),1),dtype='complex')
    tmp[:,0] = main.u
    usave = append(usave,tmp,1)
    tsave = append(tsave,main.t)


    if (live_plot == 1):
      clf()
      plot(main.k,real(main.w0hat),label='w1')
      plot(main.k,real(main.w1hat),label='w2')
      plot(main.k,real(main.w2hat),label='w3')
      legend(loc=1)
      #plot(x,u)
#      main.u = myifft(main.uhat)*sqrt(N)
#      plot(x,main.u)
      pause(0.00000001)
  itera += 1

np.savez('Solution/stats',uhat=uhatsave,t=tsave,tf=tsave_full,k = k,x=x,u=usave,tauhat=tauhatsave,Energy=main.Esave,\
         Energy_resolved=main.Esave_resolved,Dissipation=main.Dsave,Dissipation_resolved=main.Dsave_resolved)
