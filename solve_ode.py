import sys
import matplotlib.pyplot as plt
import numpy as np
from numba import jit


def euler_step(f, x1, t1, h,*args):
    x2 = x1 + h * f(t1, x1,*args)
    t2 = t1 + h
    return x2, t2


def rk4_step(f, x1, t1, h,*args):
    m1 = f(t1, x1,*args)
    m2 = f(t1 + h / 2, x1 + (h / 2) * m1,*args)
    m3 = f(t1 + h / 2, x1 + (h / 2) * m2,*args)
    m4 = f(t1 + h, x1 + h * m3,*args)
    x2 = x1 + (h / 6) * (m1 + 2 * m2 + 2 * m3 + m4)
    t2 = t1 + h
    return x2, t2


def solve_to(step,f,x1,t1,t2,hmax,*args):
    while (t1+hmax)<t2:
        x1,t1 = step(f,x1,t1,hmax,*args)
    else:
        x1,t1 = step(f,x1,t1,t2-t1,*args)
    return x1


def solve_ode(f,x0,tArr,method,hmax,system = False,*args):
    if method == "euler":
        step = euler_step
    elif method == "rk4":
        step = rk4_step
    else:
        sys.exit("Method: \"%s\" is not valid. Please select 'euler' or 'rk4'." % method)
    if system:
        sol = np.empty(shape = (len(tArr),len(x0)))
    else:
        sol = np.empty(shape=(len(tArr), 1))
    sol[0] = x0
    for i in range(len(tArr)-1):
        xi = solve_to(step,f,sol[i],tArr[i],tArr[i+1],hmax,*args)
        sol[i+1] = xi
    if system:
        outputSol = np.empty(shape = (len(x0),len(tArr)))
        for i in range(len(x0)):
            outputSol[i] = [item[i] for item in sol]
        return np.array(outputSol)
    else:
        return np.array(sol)

if __name__ == '__main__':
    def func(t, x,args):


        #print(a)
        dxdt = args*x
        return dxdt


    tArr = np.linspace(0, 1, 10)
    sol = solve_ode(func,1,tArr,'euler',0.01,False,1)
    plt.plot(tArr,sol)
    plt.show()

