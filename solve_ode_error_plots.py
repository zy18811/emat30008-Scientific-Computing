from math import pi
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
    t_array = np.linspace(0, t, 5)
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
    The tqdm function creates a progress bar in the console 
    """
    threads = cpu_count() - 1
    with Pool(threads) as p:
        error_array = list(tqdm(p.imap(err_for_h_wrapper, args), total=len(args), desc="%s" % method))
    end = timer()
    print("Time taken for %s = %f sec" % (method, end - start))
    return error_array


def main():
    """
    Calculates the error at different step sizes for the Euler and 4th order Runge-Kutta methods on the
    ODE dxdt = x at t = 1. Plots the error against step size for both methods as a loglog graph.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    hvals = np.logspace(1, -6, 1000)
    euler_errors = error_list_for_h_list("euler", hvals, 1)
    rk4_errors = error_list_for_h_list("rk4", hvals, 1)
    plotter(hvals, euler_errors, ax, "loglog", "Euler")
    plotter(hvals, rk4_errors, ax, "loglog", "RK4")
    plt.xlabel("h")
    plt.ylabel("Error")

    """
    Finds where Euler and Rk4 have the same error, and the respective step sizes. Plots one case of this as an example
    """

    ee = np.array(euler_errors)
    ee = [np.float_("%.3g" % ele) for ele in ee]
    re = np.array(rk4_errors)
    re = [np.float_("%.3g" % ele) for ele in re]

    intersect, ee_i, re_i = np.intersect1d(ee, re, return_indices=True)
    intersect_h_eul = hvals[ee_i]
    intersect_h_eul = [np.float_("%.3g" % ele) for ele in intersect_h_eul]
    intersect_h_rk4 = hvals[re_i]
    intersect_h_rk4 = [np.float_("%.3g" % ele) for ele in intersect_h_rk4]

    plt.axhline(y=intersect[5], c='k', ls='--', label=f'Error = {intersect[5]}')
    plt.axvline(x=intersect_h_eul[5], c='k', ls='--', label=f' Euler h = {intersect_h_eul[5]}', ymax=0.63)
    plt.axvline(x=intersect_h_rk4[5], c='k', ls='--', label=f'RK4 h = {intersect_h_rk4[5]}', ymax=0.63)
    plt.legend()
    plt.show()

    """
    Finds time that Euler and RK4 take for the same error using the example step size found above
    """
    euler_h = intersect_h_eul[5]
    rk4_h = intersect_h_rk4[5]

    t_array = np.linspace(0, 1, 100)

    # times Euler method
    eul_start = timer()
    solve_ode(dxdt_equals_x, 1, t_array, 'euler', euler_h, False)
    eul_end = timer()
    euler_time = eul_end - eul_start

    # time RK4 method
    rk4_start = timer()
    solve_ode(dxdt_equals_x, 1, t_array, 'rk4', rk4_h, False)
    rk4_end = timer()
    rk4_time = rk4_end - rk4_start

    print(f"Timed solving the ODE dx/dt = x\n"
          f"For the same error, the Euler method took {euler_time}s while the RK4 method took {rk4_time}s.")

    """
    d2x/dt2 = -x error
    """
    """
    The true solution of d2x/dt2 was plotted in solve_ode.main(). It is a periodic function. However, for large value
    of t, the solutions predicted by the Euler and RK4 methods diverge from the true value.
    
    This can be illustrated by plotting x against dxdt. The true solution forms a circle as it is periodic.
    """

    def d2xdt2_equals_minus_x(t, u):
        """
        Function defining system of  ODEs dxdt = y, dy/dt = -x
        :param t: t value
        :param u: array [x, y]
        :return: returns value of dxdt and dy/dt at (t,u)
        """
        x = u[0]
        y = u[1]

        dxdt = y
        dydt = -x

        return np.array([dxdt, dydt])

    def d2xdt2_equals_minus_x_true(t):
        """
        Function returning true value of system of  ODEs dxdt = y, dy/dt = -x
        :param t: t value
        :return: returns true value of x and y at t
        """
        x = np.sin(t) + np.cos(t)
        y = np.cos(t) - np.sin(t)
        return np.array([x, y])

    t = np.linspace(0, 5000, 5000)

    """
    True values
    """
    true = d2xdt2_equals_minus_x_true(np.linspace(0, 2 * pi, 100))
    true_x = true[0]
    true_y = true[1]

    """
    Euler, h = 0.01
    """
    euler_sol = solve_ode(d2xdt2_equals_minus_x, [1, 1], t, 'rk4', 0.01, True)
    euler_sol_x = euler_sol[0]
    euler_sol_y = euler_sol[1]

    """
    4th Order Runge-Kutta, h = 0.01
    """
    rk4_sol = solve_ode(d2xdt2_equals_minus_x, [1, 1], t, 'rk4', 0.01, True)
    rk4_sol_x = rk4_sol[0]
    rk4_sol_y = rk4_sol[1]

    """
    Plotting
    """

    plt.plot(euler_sol_x, euler_sol_y, label='Euler', alpha=0.5)
    plt.plot(rk4_sol_x, rk4_sol_y, label='RK4', alpha=0.5)
    plt.plot(true_x, true_y, c='r', linewidth=2, label='True')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('dx/dt')
    plt.show()

    """
    Looking at the plot, it can be seen that for large t, the solutions given by the Euler and RK4 methods diverge from 
    the true value.
    """


if __name__ == "__main__":
    main()
