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


def error(true,est):
    err = est-true
    return abs(err)


def errorPlot(method, hvals, t,format,ax):
    errArr = []
    tArr = np.linspace(0,t,1000)
    for h in hvals:
        x_est = solve_ode(func,1,tArr,h,method)[-1]
        x_true = getTrueValue(t)
        err = abs(x_est-x_true)
        errArr.append(err)
    plot(hvals,errArr,ax,format)


def main():
    #h_vals = np.linspace(5,0.0000001,100)
    h_vals = np.linspace(0.1,0.0001,20)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    errorPlot("euler",h_vals,1,"loglog",ax)
    errorPlot("rk4",h_vals,1,"loglog",ax)
    plt.legend(["Euler","RK4"])
    plt.show()


if __name__ == "__main__":
    main()



