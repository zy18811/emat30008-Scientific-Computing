import sys
import matplotlib.pyplot as plt
import numpy as np
from numba import jit


def euler_step(f, x1, t1, h):
    x2 = x1 + h * f(t1, x1)
    t2 = t1 + h
    return x2, t2


def rk4_step(f, x1, t1, h):
    m1 = f(t1, x1)
    m2 = f(t1 + h / 2, x1 + (h / 2) * m1)
    m3 = f(t1 + h / 2, x1 + (h / 2) * m2)
    m4 = f(t1 + h, x1 + h * m3)
    x2 = x1 + (h / 6) * (m1 + 2 * m2 + 2 * m3 + m4)
    t2 = t1 + h
    return x2, t2


def solve_to(step,f,x1,t1,t2,hmax):
    while (t1+hmax)<t2:
        x1,t1 = step(f,x1,t1,hmax)
    else:
        x1,t1 = step(f,x1,t1,t2-t1)
    return x1


def solve_ode(f,x0,tArr,method,hmax,system = False):
    if method == "euler":
        step = euler_step
    elif method == "rk4":
        step = rk4_step
    else:
        sys.exit("Method: \"%s\" is not valid. Please select a valid method" % method)
    if system:
        sol = np.empty(shape = (len(tArr),len(x0)))
    else:
        sol = np.empty(shape=(len(tArr), 1))
    sol[0] = x0
    for i in range(len(tArr)-1):
        xi = solve_to(step,f,sol[i],tArr[i],tArr[i+1],hmax)
        sol[i+1] = xi
    if system:
        outputSol = np.empty(shape = (len(x0),len(tArr)))
        for i in range(len(x0)):
            outputSol[i] = [item[i] for item in sol]
        return outputSol
    else:
        return sol

