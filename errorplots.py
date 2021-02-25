from solve_ode import solve_ode
from plot import plot
import numpy as np
import matplotlib.pyplot as plt


def func(t,x):
    dxdt = x
    return dxdt


def getTrueValue(t):
    x = np.exp(t)
    return x


def squaredError(true,est):
    err = true-est
    return err**2


def errorPlot(method, hvals, t,format,ax):
    errArr = []
    tArr = [0, t]
    for h in hvals:
        x_est = solve_ode(func,1,tArr,h,method)[-1]
        x_true = getTrueValue(t)
        err = squaredError(x_true,x_est)
        errArr.append(err)
    plot(hvals,errArr,ax,format)


def main():
    h_vals = np.linspace(1,0.000000001,1000)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
   # errorPlot("euler",h_vals,1,"loglog",ax)
    errorPlot("rk4",h_vals,1,"loglog",ax)
    plt.show()


if __name__ == "__main__":
   main()



