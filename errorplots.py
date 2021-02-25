from solve_ode import solve_ode
from plot import plot
import numpy as np


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


plot(hArr,errArrEul)