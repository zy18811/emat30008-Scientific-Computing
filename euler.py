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
    dydx = -0.3*x
    return dydx


t = np.linspace(0,50,200)

sol = solve_ode(func,1,t,0.1)

#true_sol = odeint(func,1,sol[0], tfirst=True)

plt.plot(t,sol)


plt.show()

