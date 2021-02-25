import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import odeint

def euler_step(f,x1,t1,h):
    x2 = x1+h*f(t1,x1)
    return x2


def solve_to(step,f,x1,t1,t2,hmax):
    t_span = t2-t1
    num_steps = math.floor(t_span/hmax)
    t_list = np.linspace(t1,t2,num_steps)
    x_list = np.zeros([num_steps])
    x_list[0] = x1
    for i in range(1,num_steps):
        x_list[i] = x_list[i-1] + hmax*f(t_list[i-1],x_list[i-1])
    return x_list[-1]  # x2


def solve_ode(f,x0,t,hmax):
    x_list = np.zeros([len(t)])
    x_list[0] = x0

    for i in range(len(t)-1):
        x = solve_to(euler_step,f,x_list[i],t[i],t[i+1],hmax)
        x_list[i+1] = x
    return x_list


def func(t,x):
    dxdt = x
    return dxdt


def getTrueValue(t):
    x = np.exp(t)
    return x


def squaredError(true,est):
    err = true-est
    return err**2


t = np.linspace(0,50,200)

#sol = solve_ode(func,1,t,0.0001)



h = 0.0001
errArr = []
hArr = []
while h < 0.1:
    true = getTrueValue(t)[-1]
    est = solve_ode(func,1,t,h)[-1]
    err = squaredError(true,est)
    errArr.append(err)
    hArr.append(h)
    h+=0.0001

plt.plot(hArr,errArr)


#plt.plot(t,true_vals)

#plt.plot(t,sol)


plt.show()

