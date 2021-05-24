import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve, root
from tqdm import tqdm

from shooting import shootingG

"""
Functions implementing natural parameter and pseudo-arclength numerical continuation. Finds how the solutions of a 
function change as a parameter is varied, allowing a phase portrait to be plotted. 
"""


def continuation(method, function, u0, pars, vary_par, vary_par_range, vary_par_number=100, discretisation=shootingG,
                 solver=fsolve, pc=None):
    """
    Takes a function and applies a specified continuation method over a range of parameter values for a selected
    parameter. Returns the solution of the function for each parameter value.

    :param method: Numerical continuation method to use. 'natural' for natural parameter continuation and 'pseudo' for
    pseudo-arclength continuation.
    :param function: Function to vary the parameter of
    :param u0: Array containing guess for initial state of the function
    :param pars: Array containing the function parameters
    :param vary_par: Index of the parameter to vary in pars
    :param vary_par_range: Array specifying the inclusive range to vary the parameter in the form [a, b] where a is the
    lower bound and b is the upper bound.
    :param vary_par_number: Number of equally spaced values to split the range into
    :param discretisation: Discretisation to be applied to function. Use shootingG for shooting problems
    :param solver: Solver to use for root finding. fsolve is suggested as it has the best performance
    :param pc: Phase condition function for shooting problems
    :return: Returns par_list, x where par_list is a list of parameters and sols is a list of the respective function
    solutions.
    """

    def sol_wrapper(val, u0):
        """
        Takes a parameter value and a function state and returns the solution
        :param val: Parameter value
        :param u0: Function state
        :return: Array containing the solution of the function for the given parameter value and function state
        """
        pars[vary_par] = val    # sets function parameters with the given value

        # Adds the phase condition to the function arguments if one is given
        if pc is None:
            args = pars
        else:
            args = (pc, pars)

        # Solves the function with the given discretisation with the specified function state and parameter value and
        # returns the solution
        return np.array(solver(discretisation(function), u0, args=args))

    if method == 'natural':
        par_list, x = natural_parameter_continuation(u0, vary_par_range, vary_par_number, sol_wrapper)
    elif method == 'pseudo':
        par_list, x = pseudo_arclength_continuation(function, u0, pars, vary_par, vary_par_range, vary_par_number,
                                                    discretisation, pc, sol_wrapper)
    else:
        raise ValueError(f"method: '{method}' is not valid. Please select 'natural' or 'pseudo'.")
    normed_x = np.linalg.norm(np.delete(x, -1, axis=1), axis=1)
    plt.plot(par_list, x[:, 0])
    return par_list, x


def natural_parameter_continuation(u0, vary_par_range, vary_par_number, solWrapper):
    """
    Natural parameter numerical continuation. Takes an initial state and a range of parameters and returns the solution
    of the function for a parameter varied in the range. The solution for a given parameter is used as the initial
    state for the next parameter. The parameters are equally spaced in the range and are iterated over from one end to
    the other.

    :param u0: Initial guess of function state
    :param vary_par_range: Array specifying the inclusive range to vary the parameter in the form [a, b] where a is the
    lower bound and b is the upper bound.
    :param vary_par_number: Number of equally spaced values to split the range into
    :param solWrapper: Returns the value of the function for a particular parameter value and function state
    :return: Returns par_list, sols where par_list is the list of parameter values iterated over and sols is an array
    of the solutions for each parameter value.
    """

    # generates parArr using vary_par_range and vary_par_number
    par_list = np.linspace(vary_par_range[0], vary_par_range[1], vary_par_number)

    # iterates over each value in parArr and returns the function solution. This is then used as the function state for
    # the next iteration
    sols = []
    for val in tqdm(par_list):
        u0 = solWrapper(val, u0)
        sols.append(u0)
        u0 = np.round(u0, 2)
    sols = np.array(sols)
    return par_list, sols


def pseudo_arclength_continuation(function, u0, pars, vary_par, vary_par_range, vary_par_number, discretisation, pc,
                                  solWrapper):
    """
    Pseudo-arclength continuation. Takes an initial function state and a parameter range and returns the solution
    of the function for a parameter varied in the range. The solution for a given parameter is used as the initial
    state for the next parameter. The parameters are equally spaced in the range and are iterated over from one end to
    the other. Starting with the initial parameter values, the subsequent values are determined by the solution of the
    pseudo-arclength equation.

    :param function: Function to vary the parameter of
    :param u0: Array containing guess for initial state of the function
    :param pars: Array containing the function parameters
    :param vary_par: Index of the parameter to vary in pars
    :param vary_par_range: Array specifying the inclusive range to vary the parameter in the form [a, b] where a is the
    lower bound and b is the upper bound.
    :param vary_par_number: Number of equally spaced values to split the range into
    :param discretisation: Discretisation to be applied to function. Use shootingG for shooting problems
    :param pc: Phase condition function for shooting problems
    :param solWrapper: Returns the value of the function for a particular parameter value and function state
    :return: Returns par_list, sols where par_list is the list of parameter values iterated over and sols is an array
    of the solutions for each parameter value.
    """

    def pseudoGetTwoTrue(u0, dp):
        """
        Gets the two true values needed to start the pseudo-arclength method.
        :param u0: Array containing guess for initial state of the function
        :param dp: Difference in parameter for the two true values. Determines the approximate step between parameter
        values
        :return: Returns true0, true1 where these are two true values for initial parameter values.
        """

        # Sets sign of dp depending on whether the parameter is increasing or decreasing as it moves through the range
        dp *= np.sign(vary_par_range[1] - vary_par_range[0])
        p0 = vary_par_range[0]
        p1 = p0 + dp
        true0 = np.append(solWrapper(p0, u0), p0)
        true1 = np.append(solWrapper(p1, np.round(true0[:-1], 2)), p1)
        return true0, true1

    def pseudo(x, delta_x, p, delta_p):
        """
        Pseudo-arclength equation. Takes state vector (x), secant of the state vector (delta_x), parameter value (p),
        and secant of the parameter value (delta_p), returns the value of the pseudo-arclength equation for these values

        :param x: Function solution (state vector)
        :param delta_x: Secant of the function solution (state vector)
        :param p: Parameter value
        :param delta_p: Secant of the parameter value
        :return: Returns the value of the pseudo-arclength equation for the given values
        """

        # calculates predicted values of x and p by adding the secant
        x_pred = x + delta_x
        p_pred = p + delta_p
        #ds = np.linalg.norm(np.append(delta_x, delta_p))
        # pseudo-arclength equation
        arc = np.dot(x - x_pred, delta_x) + np.dot(p - p_pred, delta_p)# - ds
        return arc

    def F(x, function, pc, discretisation, delta_x, delta_p, pars, vary_par):
        """
        Augmented root finding problem - original root finding problem plus the pseudo-arclength equation.
        :param x: State vector
        :param function: Function to vary the parameter of
        :param pc: Phase condition (if shooting problem)
        :param discretisation: Discretisation to be applied to function. Use shootingG for shooting problems
        :param delta_x: Secant of the state vector
        :param delta_p: Secant of the parameter value
        :param pars: Array containing the function parameters
        :param vary_par: Index of the parameter to vary in pars
        :return: Returns value of augmented root finding function for given arguments
        """
        u0 = x[:-1]  # + delta_x
        p = x[-1]  # + delta_p
        pars[vary_par] = p  # sets function parameters with the given value
        d = discretisation(function)

        # value of initial root finding problem for given arguments
        if pc is None:
            g = d(u0, pars)
        else:
            g = d(u0, pc, pars)
        arc = pseudo(u0, delta_x, p, delta_p)   # value of pseudo-arclength function for given arguments

        # appends pseudo-arclength value to initial root finding value to create the augmented root finding value
        f = np.append(g, arc)
        return f

    v0, v1 = pseudoGetTwoTrue(u0, 0.05) # two initial true values

    # plt.plot(v0[-1], v0[0], 'r+')
    # plt.plot(v1[-1], v1[0], 'r+')

    sols = []
    par_list = []

    while True:
        delta_x = v1[:-1] - v0[:-1]
        delta_p = v1[-1] - v0[-1]

        pred_v_x = v1[:-1] + delta_x
        pred_v_p = v1[-1] + delta_p

        pred_v = np.append(pred_v_x, pred_v_p)
        pars[vary_par] = pred_v[-1]
        # plt.plot(u[-1],u[0],'g*')

        # solves augmented root finding problem for given arguments
        solution = root(F, pred_v, method='lm', args=(function, pc, discretisation, delta_x, delta_p, pars, 0))
        sol = solution['x']

        if sol[0] < 0:
            break

        # appends solution and respective parameter value to lists to be returned
        sols.append(sol[:-1])
        par_list.append(sol[-1])

        # plt.plot(sol[-1], sol[0], 'gx')

        # plt.axline((u[-1],u[0]),(sol[-1],sol[0]),c = 'k',ls = '--',alpha = 0.4)

        v0 = np.round(v1, 2)
        v1 = np.round(sol, 2)

    # converts sols to an ndarray to allow better slicing
    sols = np.array(sols)
    return par_list, sols


def hopfNormal(t, u, args):
    beta = args[0]
    sigma = args[1]

    u1 = u[0]
    u2 = u[1]
    du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
    if du1dt == np.NaN:
        print(u1, u2, beta)

    du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)
    return np.array([du1dt, du2dt])


def pcHopfNormal(u0, args):
    p = hopfNormal(1, u0, args)[0]
    return p


def modHopfNormal(t, u, args):
    beta = args[0]

    u1 = u[0]
    u2 = u[1]

    du1dt = beta * u1 - u2 + u1 * (u1 ** 2 + u2 ** 2) - u1 * (u1 ** 2 + u2 ** 2) ** 2
    du2dt = u1 + beta * u2 + u2 * (u1 ** 2 + u2 ** 2) - u2 * (u1 ** 2 + u2 ** 2) ** 2
    return np.array([du1dt, du2dt])


def pcModHopfNormal(u0, args):
    p = modHopfNormal(1, u0, args)[0]
    return p


def cubic(x, c):
    return x ** 3 - x + c


u0_hopfNormal = np.array([1.4, 0, 6.3])

# continuation('natural',hopfNormal,u0_hopfNormal,[2, -1],0,[2,-1],100,shootingG,fsolve,pcHopfNormal)
# continuation('pseudo',hopfNormal,u0_hopfNormal,[2,-1],0,[2,-1],200,shootingG,fsolve,pcHopfNormal)

continuation('natural', modHopfNormal, u0_hopfNormal, [2], 0, [2, -1], 50, shootingG, fsolve, pcModHopfNormal)
continuation('pseudo', modHopfNormal, u0_hopfNormal, [2], 0, [2, -1], 30, shootingG, fsolve, pcModHopfNormal)

# continuation('natural',cubic,[1,1,1],[-2],0,[-2,2],200,discretisation= lambda x:x,solver=fsolve,pc=None)
# continuation('pseudo',cubic,np.array([1,1,1]),[2],0,[-2,2],200,lambda x:x,fsolve,None)


plt.grid()
plt.show()
