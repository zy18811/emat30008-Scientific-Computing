from solve_ode import solve_ode
from plot import plot
import numpy as np
import matplotlib.pyplot as plt
import sys


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
    x_true = getTrueValue(t)
    errArr=[]
    tArr = np.linspace(0,t,100)
    for h in hvals:
        x_est = solve_ode(func,1,tArr,method,h)
        err = error(x_true,x_est)
        errArr.append(err)
    if format == "linear":
        ax.plot(hvals,errArr)
    elif format == "loglog":
        ax.loglog(hvals,errArr)
    else:
        sys.exit("Format: \"%s\" is not valid. Please select a valid format for plotting." % format)




def main():
    hvals = np.linspace(1,0.00001,1000)
    errorPlot("euler",hvals,1,"loglog",ax)


if __name__ == "__main__":
    main()



