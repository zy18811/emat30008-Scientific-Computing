import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import odeint

def euler_step(func,x_n,t_n,h):
    x_n_plus_1 = x_n + h*func(t_n,x_n)
    return x_n_plus_1

def solve_to(func, x1,t1,x2,t2,h):
    n_space = range(math.floor(t2/h))
    tn = t1
    xn = x1
    x_arr = []
    for n in n_space:
        x_arr.append(xn)
        xn = euler_step(func,xn,tn,h)
        tn = tn + h
    x_arr.append(x2)
    return x_arr

def func(t,y):
    k = 0.3
    dydt = -k*y
    return dydt


x1 = 20
x2 = 0
t1 = 0
t2 = 20
ans = solve_to(func,x1,t1,x2,t2,0.1)
y = odeint(func, 20, [0,20])
plt.plot(y)
plt.plot(np.linspace(t1,t2,len(ans)),ans)
plt.show()

