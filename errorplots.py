from solve_ode import solve_ode
from plotter import plotter
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from timeit import default_timer as timer
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


def errForH(method,h,t):
    tArr = np.linspace(0, t, 10)
    x_est = solve_ode(func,1,tArr,method,h)[-1]
    x_true = getTrueValue(t)
    err = error(x_true,x_est)
    return err


def errForHWrapper(args):
    return errForH(*args)


def errorPlot(method, hvals, t,format,ax):

    args = [(method,h,t) for h in hvals]

    start = timer()
    threads = cpu_count() - 1
    with Pool(threads) as p:
        errArr = list(tqdm(p.imap(errForHWrapper,args),total = len(args),desc = "%s" % method))
    end = timer()
    print("Time taken for %s = %f sec" % (method,end-start))
    plotter(hvals,errArr,ax,format,method)


def main():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    hvals = np.logspace(1,-4,100)
    errorPlot("euler",hvals,1,"loglog",ax)
    errorPlot("rk4",hvals,1,"loglog",ax)
    plt.xlabel("h")
    plt.ylabel("Error")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()



