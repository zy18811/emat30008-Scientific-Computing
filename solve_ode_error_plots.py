from multiprocessing import Pool, cpu_count
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from solve_ode import solve_ode


"""
Error plots for comparing performance of Euler and 4th order Runge-Kutta methods.
"""


def plotter(x, y, ax, plot_format, label):
    """
    Plots data in either loglog or linear format
    :param x: x data
    :param y: y data
    :param ax: matplotlib ax object to plot on
    :param plot_format: plot format - either 'loglog' or 'linear'
    :param label: figure label
    """
    if plot_format == "loglog":
        ax.loglog(x, y, label=label)
    elif plot_format == "linear":
        ax.plot(x, y, label=label)
    else:
        raise ValueError(f"format: '{plot_format}' is not valid. Please select 'loglog' or 'linear'.")


def dxdt_equals_x(t, x):
    """
    Function defining ODE dxdt = x
    :param t: t value
    :param x: x value
    :return: returns value of dxdt at (t,x)
    """
    dxdt = x
    return dxdt


def dxdt_equals_x_true(t):
    """
    Returns true values of x for the ODE dxdt = x for given values of t
    :param t: t value(s) to return solution for
    :return: Returns true values of x for the ODE dxdt = x for given values of t
    """
    x = np.exp(t)
    return x


def error(true, est):
    """
    Returns the absolute error between a true value, true, and an estimated value, est
    """
    err = est - true
    return abs(err)


def dxdt_equals_x_err_for_h(method, h, t):
    """
    Gets error between the true estimated solution of dxdt = x at time t, for a given method with step size h
    :param method: Solution estimator, "euler" for Euler method, "rk4" for 4th order Runge-Kutta method
    :param h: Step size
    :param t: Time value to get solution for
    :return: Returns error of method at t with given h
    """
    t_array = np.linspace(0, t, 10)
    x_est = solve_ode(dxdt_equals_x, 1, t_array, method, h, False)[-1]  # value of estimated solution at given t
    x_true = dxdt_equals_x_true(t)  # true value at given t
    err = error(x_true, x_est)  # error between true and estimated solution
    return err


def err_for_h_wrapper(args):
    """
    Wrapper function for dxdt_equals_x_err_for_h() to allow multiprocessing
    :param args: args = (method,h,t)
    :return: Returns value of dxdt_equals_x_err_for_h() for given arguments
    """
    return dxdt_equals_x_err_for_h(*args)


def error_list_for_h_list(method, hvals, t):
    """
    Takes a method, a list of step size values, and a time value. Returns the error of the specified method at the given
    time value for each step size.
    :param method:  Solution estimator, "euler" for Euler method, "rk4" for 4th order Runge-Kutta method
    :param hvals: List of step sizes to return error for
    :param t: t value to calulate error at
    :return: Returns a list of errors. The position of each each error is the same as the index of the corresponding
    step size in hvals
    """
    args = [(method, h, t) for h in hvals]  # generates args for err_for_h_wrapper()
    start = timer()

    """
    Uses multiprocessing package to speed up computation of error for very small step sizes
    """
    threads = cpu_count()
    with Pool(threads) as p:
        error_array = list(tqdm(p.imap(err_for_h_wrapper, args), total=len(args), desc="%s" % method))
    end = timer()
    print("Time taken for %s = %f sec" % (method, end - start))
    return error_array


def main():
    """
    Calculates the error at different step sizes for the Euler and 4th order Runge-Kutta methods on the
    ODE dxdt = x at t = 1. Plots the error against step size for both methods.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    hvals = np.logspace(1, -8, 1000)
    euler_errors = error_list_for_h_list("euler", hvals, 1)
    rk4_errors = error_list_for_h_list("rk4", hvals, 1)
    plotter(hvals, euler_errors, ax, "loglog", "Euler")
    plotter(hvals, rk4_errors, ax, "loglog", "RK4")
    plt.xlabel("h")
    plt.ylabel("Error")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
