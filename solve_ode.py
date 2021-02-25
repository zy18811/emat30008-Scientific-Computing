import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import odeint
import sys

def euler_step(f,x1,t1,h):
    x2 = x1+h*f(t1,x1)
    t2 = t1+h
    return t2,x2


def rk4_step(f,x1,t1,h):
    m1 = f(t1,x1)
    m2 = f(t1+h/2,x1+(h/2)*m1)
    m3 = f(t1+h/2,x1+(h/2)*m2)
    m4 = f(t1+h,x1+h*m3)
    x2 = x1+(h/6)*(m1+2*m2+2*m3+m4)
    t2 = t1+h
    return t2,x2


def solve_to(step,f,x1,t1,t2,hmax):
    while t1 <= t2:
        (t1,x1)= step(f,x1,t1,hmax)
    if t1<t2:
        (t1,x1) = step(f,x1,t1,t2-t1)
    return x1




def solve_ode(f,x0,t,hmax,method):
    if method == "rk4":
        step = rk4_step
    elif method == "euler":
        step = euler_step
    else:
        sys.exit("Method: \"%s\" is not valid. Please select a valid method" % method)
    x_sol = []
    for i in range(len(t)-1):
        x_sol.append(x0)
        x0 = solve_to(step,f,x0,t[i],t[i+1],hmax)
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


t = np.linspace(0,1,100)


hArr = []
errArrEul = []
errArrRK4 = []
for i in range(7):
    h = 10**-i
    print("h = %f" % h)
    x1_est_eul = solve_ode(func,1,t,h,"euler")[-1]
    x1_est_rk4 = solve_ode(func,1,t,h,"rk4")[-1]
    x1_true = getTrueValue(1)
    errEul = squaredError(x1_true,x1_est_eul)
    errArrEul.append(errEul)
    errEul = squaredError(x1_true, x1_est_rk4)
    errArrRK4.append(errEul)
    hArr.append(h)


plt.plot(np.log(hArr),np.log(errArrEul))
plt.plot(np.log(hArr),np.log(errArrRK4))
plt.show()

