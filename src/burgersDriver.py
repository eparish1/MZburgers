from pylab import *
from padding import *
import numpy as np
from myffts import *
from RHSfunctions import *
from Classes import *
close("all")
if not os.path.exists('Solution'):
   os.makedirs('Solution')

def advanceQ_RK4(main):
  Q0 = zeros(size(main.Q),dtype='complex')
  Q0[:] = main.Q[:]
  rk4const = array([1./4,1./3,1./2,1.])
  if (main.turb_model == 99):
    w0 = main.computeW(main.uhat,kc)
  for i in range(0,4):
    main.computeRHS(main)
    main.Q = Q0 + main.dt*rk4const[i]*main.RHS
    main.Q2U()
  if (main.turb_model == 99):
    wf = main.computeW(main.uhat,kc)
    main.wdot = (wf - w0)/main.dt

uhat[-1] = 0
main = variables(turb_model,N,k,uhat,u,t,kc,dt,nu,x) 

ion()
itera = 0
w0Esave = zeros(1)

while (main.t <= et):
  main.itera = itera
  main.U2Q()
  advanceQ_RK4(main)
  main.t += dt
  print(main.t)
  main.saveHook1()
  if (itera%save_freq == 0):
    main.saveHook2()
    if (live_plot == 1):
      clf()
      plot(x,u)
      main.u = myifft(main.uhat)*sqrt(N)
      plot(x,main.u)
      pause(0.00000001)
  itera += 1

main.saveHookFinal()
