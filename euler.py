import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import odeint

def euler_step(f,x1,t1,h):
    x2 = x1+h*f(t1,x1)
    t2 = t1+h
    return t2,x2


def solve_to(step,f,x1,t1,t2,hmax):
    while t1 <= t2:
        (t1,x1)=step(f,x1,t1,hmax)
    if t1<t2:
        (t1,x1) = step(f,x1,t1,t2-t1)
    return x1




def solve_ode(f,x0,t,hmax):
    x_sol = []
    for i in range(len(t)-1):
        x_sol.append(x0)
        x0 = solve_to(euler_step,f,x0,t[i],t[i+1],hmax)
    x_sol.append(x0)
    return x_sol




def func(t,x):
    dxdt = x
    return dxdt


def getTrueValue(t):
    x = np.exp(t)
    return x


def squaredError(true,est):
    err = true-est
    return err**2


t = np.linspace(0,10,100)

sol = solve_ode(func,1,t,0.00001)

true_vals = getTrueValue(t)

plt.plot(t,true_vals)

plt.plot(t,sol)


plt.show()

