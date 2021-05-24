from solve_ode import solve_ode
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from newtonrhapson import newton
from periodfinderforcheck import manual_period_finder
from shooting import orbitShooting

def predator_prey(t,y,args):

    x = y[0]
    y = y[1]

    a = args[0]
    d = args[1]
    b = args[2]

    dxdt = x*(1-x) - (a*x*y)/(d+x)
    dydt = b*y*(1-(y/x))
    return np.array([dxdt,dydt])


def pc(u0,args):
    p = predator_prey(1,u0,args)[0]
    return p


x0 = np.array([0.5,0.5,15])



t = np.linspace(0,1000,10000)
sol = solve_ode(predator_prey,np.array([0.25,0.25]),t,"rk4",0.001,True,[1,0.1,0.16])

xsol = sol[0]
ysol = sol[1]

plt.plot(xsol,ysol)
plt.legend()


'''
valFind = xvalyval(t,xeul,yeul,dp = 6)
print(f"valfind = {valFind}")
'''

'''
xval_newt = newt[0]
yval_newt = newt[1]

xval_fsolve = fsolve[0]
yval_fsolve = fsolve[1]

xval_find = valFind[0]
yval_find = valFind[1]
'''

#fsolve = fsolve(G,x0,args = (f,pc))
orbit = orbitShooting(predator_prey,x0,pc,fsolve,[1,0.1,0.16])
plt.plot(orbit[0],orbit[1],'r+')
#plt.plot(xval_newt,yval_newt,'r+',label = "newt")
#plt.plot(xval_fsolve,yval_fsolve,'b1',label = "fsolve")
#plt.plot(xval_find,yval_find,'gx',label = "find")


plt.legend()


plt.show()
