from solve_ode import solve_ode
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from newtonrhapson import newton
from periodfinderforcheck import xvalyval
from shooting import orbitShooting

def func(t,y):
    x = y[0]
    y = y[1]
    a = 1
    d = 0.1
    b = 0.16
    dxdt = x*(1-x) - (a*x*y)/(d+x)
    dydt = b*y*(1-(y/x))
    return np.array([dxdt,dydt])


def G(x,*args): # x = [u0,T], args = (f,phase)
    u01 = x[0]
    u02 = x[1]
    T = x[2]
    u0 = [u01,u02]
    f = args[0]
    phase = args[1]

    sol = f(u0,T)

    g = [u0[0]-sol[0],u0[1]-sol[1]]

    p = phase(u0)
    return np.array([g[0],g[1],p])



def f(u0,T):
    tArr = np.linspace(0,T,1000)
    sol = solve_ode(func, u0, tArr, "rk4",0.01,system=True)
    return np.array([sol[0][-1],sol[1][-1]])


def pc(u0):
    x = u0[0]
    y = u0[1]
    a = 1
    d = 0.1
    b = 0.1
    p = x*(1-x) - (a*x*y)/(d+x)
    return p


x0 = np.array([0.5,0.5,15])

'''
args = (f,pc)
newt = newton(G,x0,pc)
print(f"newt ={newt}")


print(f"fsolve = {fsolve}")
'''

t = np.linspace(0,1000,10000)
eulsol = solve_ode(func,np.array([0.25,0.25]),t,"rk4",0.001,system=True)
xeul = eulsol[0]
yeul = eulsol[1]

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

fsolve = fsolve(G,x0,args = (f,pc))
orbit = orbitShooting(func,x0,pc)
print(fsolve)
print(orbit)
plt.plot(orbit[0],orbit[1],'r+')
#plt.plot(xval_newt,yval_newt,'r+',label = "newt")
#plt.plot(xval_fsolve,yval_fsolve,'b1',label = "fsolve")
#plt.plot(xval_find,yval_find,'gx',label = "find")

plt.plot(xeul,yeul)
plt.legend()


plt.show()
