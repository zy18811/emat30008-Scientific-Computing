import warnings
from math import pi
from shooting import shootingG
import scipy
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve, root
from tqdm import tqdm

from solve_ode import integer_float_array_input_check
from solve_pde import solve_diffusive_pde

"""
Functions implementing natural parameter and pseudo-arclength numerical continuation. Finds how the solutions of a 
function change as a parameter is varied, allowing a phase portrait to be plotted. 
"""


def continuation(method, function, u0, pars, vary_par, vary_par_range, vary_par_number, discretisation, solver=fsolve,
                 pc=None):
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
    :return: Returns par_list, x where par_list is a list of parameters and x is a list of the respective discretised
    function solutions.
    """
    # ignores warnings from numpy and scipy as there are some issues but results are not affected
    warnings.filterwarnings('ignore')
    """
    checks type of u0 - see solve_ode.py for function details
    """
    integer_float_array_input_check("u0", u0)

    """
    checks type of pars - see solve_ode.py for function details
    """
    integer_float_array_input_check("pars", pars)

    """
    checks that vary_par is an integer >=0, raises a Type/ValueError if not
    """
    if not isinstance(vary_par, (int, np.int_)):
        raise TypeError(f"vary_par: {vary_par} is not an integer")
    else:
        if vary_par < 0:
            raise ValueError(f"vary_par: {vary_par} is < 0")

    """
    checks that vary_par_number is an integer >=1, raises a Type/ValueError is not
    """
    if not isinstance(vary_par_number, (int, np.int_)):
        raise TypeError(f"vary_par_number: {vary_par_number} is not an integer")
    else:
        if vary_par_number < 1:
            raise ValueError(f"vary_par: {vary_par} is < 1")

    """
    checks that vary_par_range is a list or ndarray in the form [a,b] where a and b are integers of floats
    """
    if isinstance(vary_par_range, (list, np.ndarray)):
        if len(vary_par_range) == 2:
            integer_float_array_input_check("vary_par_range", vary_par_range)
        else:
            raise ValueError(f"vary_par_range: {vary_par_range} must be in the shape [a, b].")
    else:
        raise TypeError(f"vary_par_range: {vary_par_range} is not a list or ndarry.")

    """
    If pc isn't None, checks if pc is a function. If it is checks if it returns a scalar output. Raises an error if not
    """
    if pc is not None:
        if callable(pc):

            # tests that phase condition output is an int or float
            test = pc(u0, pars)
            if not isinstance(test, (int, float, np.int_, np.float_)):
                raise TypeError(
                    f"Output of phase condition is {type(test)}. Output needs to be of type int or float")
        else:
            raise TypeError(f"pc: '{pc}' needs to be a function.")

    """
    checks if 'function' param is a function. If it is it checks that that discretisation(function) is a 
    function. If it is also, checks that the output of discretisation(function) is of the right shape and type.
    """
    if callable(function):
        if callable(discretisation(function)):
            d = discretisation(function)
            if pc is None:
                test = d(u0, pars)
            else:
                test = d(u0, pc, pars)

            # tests that discretisation(function) output is an int, float, list or ndarray
            if isinstance(test, (int, np.int_, np.float_, list, np.ndarray)):
                # tests that function output has same shape as u0
                if not np.array(test).shape == np.array(u0).shape:
                    raise ValueError("Shape mismatch. Shape of u0 and discretisation(function) output not the same")
            else:
                raise TypeError(f"Output of discretisation(function) is {type(test)}. Output needs to be of type int, "
                                f"float, list or ndarray")
        else:
            raise TypeError(f"discretisation(function): the discretisation of function needs to be a function.")
    else:
        raise TypeError(f"function: '{function}' needs to be a function.")

    def sol_wrapper(val, u0):
        """
        Takes a parameter value and a function state and returns the solution
        :param val: Parameter value
        :param u0: Function state
        :return: Array containing the solution of the function for the given parameter value and function state
        """
        pars[vary_par] = val  # sets function parameters with the given value

        # Adds the phase condition to the function arguments if one is given
        if pc is None:
            args = pars
        else:
            args = (pc, pars)
        # Solves the function with the given discretisation with the specified function state and parameter value and
        # returns the solution
        return np.array(solver(discretisation(function), u0, args=args))

    """"
    checks if method param is valid, raises ValueError if not
    """
    if method == 'natural':
        par_list, x = natural_parameter_continuation(u0, vary_par_range, vary_par_number, sol_wrapper)
    elif method == 'pseudo':
        par_list, x = pseudo_arclength_continuation(function, u0, pars, vary_par, vary_par_range, vary_par_number,
                                                    discretisation, pc, sol_wrapper)
    else:
        raise ValueError(f"method: '{method}' is not valid. Please select 'natural' or 'pseudo'.")

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
    # the next iteration. tqdm() prints a progress bar to the console.
    sols = []
    for val in tqdm(par_list, desc="Nat. Param. Cont.: Iterating through parameter values"):
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

        # pseudo-arclength equation
        arc = np.dot(x - x_pred, delta_x) + np.dot(p - p_pred, delta_p)
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
        u0 = x[:-1]
        p = x[-1]
        pars[vary_par] = p  # sets function parameters with the given value
        d = discretisation(function)

        # value of initial root finding problem for given arguments
        if pc is None:
            g = d(u0, pars)
        else:
            g = d(u0, pc, pars)
        arc = pseudo(u0, delta_x, p, delta_p)  # value of pseudo-arclength function for given arguments

        # appends pseudo-arclength value to initial root finding value to create the augmented root finding value
        f = np.append(g, arc)
        return f

    v0, v1 = pseudoGetTwoTrue(u0, 0.05)  # two initial true values

    sols = []
    par_list = []

    # tqdm tracks iterations in the console
    with tqdm() as pbar:
        pbar.set_description("Pseudo Arc-length Cont.: Iterating")
        # iterates until halt condition
        while True:
            pbar.update(1)
            delta_x = v1[:-1] - v0[:-1]
            delta_p = v1[-1] - v0[-1]

            pred_v_x = v1[:-1] + delta_x
            pred_v_p = v1[-1] + delta_p

            pred_v = np.append(pred_v_x, pred_v_p)
            pars[vary_par] = pred_v[-1]

            # solves augmented root finding problem for given arguments
            solution = root(F, pred_v, method='lm', args=(function, pc, discretisation, delta_x, delta_p, pars, 0))
            sol = solution['x']

            normed_sol = np.linalg.norm(sol[:-1])
            normed_v1 = np.linalg.norm(v1[:-1])

            # halts when sol would cross the y-axis
            if normed_sol - normed_v1 > 0:
                pbar.close()
                break

            # appends solution and respective parameter value to lists to be returned
            sols.append(sol[:-1])
            par_list.append(sol[-1])

            v0 = v1
            v1 = sol

    # converts sols to an ndarray to allow better slicing
    sols = np.array(sols)
    return par_list, sols


if __name__ == '__main__':
    """
    Example 1 algebraic cubic equation
    """
    """
    function for cubic equation
    """


    def cubic(x, args):
        c = args[0]
        return x ** 3 - x + c


    """
    Natural parameter continuation - varying c from -2 to 2
    """
    par_list_nat, x_nat = continuation('natural', cubic, np.array([1, 1, 1]), [2], 0, [-2, 2], 200,
                                       discretisation=lambda x: x, solver=fsolve, pc=None)

    """
    Pseudo-arclength continuation - varying c from -2 to 2
    """
    par_list_ps,x_ps = continuation('pseudo',cubic,np.array([1,1,1]),[2],0,[-2,2],200,lambda x:x,fsolve,None)

    """
    Plotting c against ||x|| for both continuation methods
    """
    norm_x_nat = scipy.linalg.norm(x_nat, axis = 1, keepdims = True)
    norm_x_ps = scipy.linalg.norm(x_ps, axis = 1, keepdims = True)
    plt.plot(par_list_nat, norm_x_nat[:, 0],label = 'Nat. Param.')
    plt.plot(par_list_ps, norm_x_ps[:, 0],label = 'Pseudo-arclength')
    plt.xlabel('c')
    plt.ylabel('||x||')
    plt.legend()
    plt.show()

    """
    It can be seen from the figure that there is a fold at around c == 0.4.
    Natural parameter continuation fails after the fold as expected.
    Pseudo-arclength continuation succeeds and follows the curve around the fold.
    """

    """
    Example 2: Hopf normal form equations - shooting
    """
    """
    function for hopf normal form 
    """
    def hopfNormal(t, u, args):
        beta = args[0]
        sigma = args[1]

        u1 = u[0]
        u2 = u[1]

        du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)
        return np.array([du1dt, du2dt])

    """
    Phase condition for hopf normal form
    """
    def pcHopfNormal(u0, args):
        p = hopfNormal(1, u0, args)[0]
        return p

    """
    initial values for hopf normal form
    """
    u0_hopfNormal = np.array([1.4, 0, 6.3])

    """
    Natural parameter continuation - varying beta from 2 to -1
    """

    par_list_nat, x_nat = continuation('natural', hopfNormal, u0_hopfNormal, [2, -1], 0, [2, -1], 30, shootingG, fsolve,
                                       pcHopfNormal)

    """
    Pseudo arclength continuation - varying beta from 2 to -1
    """
    par_list_ps, x_ps = continuation('pseudo', hopfNormal, u0_hopfNormal, [2, -1], 0, [2, -1], 200, shootingG, fsolve,
                                     pcHopfNormal)

    """
    Plotting beta against ||x|| for both continuation methods
    """
    # excluding T from x
    norm_x_nat = scipy.linalg.norm(x_nat[:,:-1], axis=1, keepdims=True)
    norm_x_ps = scipy.linalg.norm(x_ps[:,:-1], axis=1, keepdims=True)
    plt.plot(par_list_nat, norm_x_nat[:, 0], label='Nat. Param.')
    plt.plot(par_list_ps, norm_x_ps[:, 0], label='Pseudo-arclength')
    plt.xlabel('beta')
    plt.ylabel('||x||')
    plt.legend()
    plt.show()

    """
    It can be seen there is a fold at beta = 0.
    Natural parameter continuation fails after the fold as expected.
    Pseudo-arclength continuation succeeds and follows the curve around the fold.
    """

    """
    Example 3: modified Hopf normal equations
    """

    """
    function for modified hopf normal
    """
    def modHopfNormal(t, u, args):
        beta = args[0]

        u1 = u[0]
        u2 = u[1]

        du1dt = beta * u1 - u2 + u1 * (u1 ** 2 + u2 ** 2) - u1 * (u1 ** 2 + u2 ** 2) ** 2
        du2dt = u1 + beta * u2 + u2 * (u1 ** 2 + u2 ** 2) - u2 * (u1 ** 2 + u2 ** 2) ** 2
        return np.array([du1dt, du2dt])

    """
    phase condition for modified hopf normal
    """
    def pcModHopfNormal(u0, args):
        p = modHopfNormal(1, u0, args)[0]
        return p

    """
    using same u0 as hopf normal
    """

    """
    Natural parameter continuation varying beta from 2 to -1
    """
    par_list_nat, x_nat = continuation('natural', modHopfNormal, u0_hopfNormal, [2], 0, [2, -1], 50, shootingG, fsolve,
                                       pcModHopfNormal)

    """
    Pseuo-arclength continuation varying beta from 2 to -1
    """
    par_list_ps, x_ps = continuation('pseudo', modHopfNormal, u0_hopfNormal, [2], 0, [2, -1], 30, shootingG, fsolve,
                                     pcModHopfNormal)

    """
    Plotting beta against ||x|| for both continuation methods
    """
    # excluding T from x
    norm_x_nat = scipy.linalg.norm(x_nat[:,:-1], axis=1, keepdims=True)
    norm_x_ps = scipy.linalg.norm(x_ps[:,:-1], axis=1, keepdims=True)
    plt.plot(par_list_nat, norm_x_nat[:, 0], label='Nat. Param.')
    plt.plot(par_list_ps, norm_x_ps[:, 0], label='Pseudo-arclength')
    plt.xlabel('beta')
    plt.ylabel('||x||')
    plt.legend()
    plt.show()

    """
    It can be seen there is a fold at around beta = -0.25
    Natural parameter continuation fails after the fold as expected.
    Pseudo-arclength continuation succeeds and follows the curve around the fold.
    """

    """
    Example 4: Numerical Continuation on a PDE
    See solve_pde.py for more information on solve_diffusive_pde() and related functions
    """
    """
    The continuation() function can be used to perform numerical continuation on diffusive PDEs in order to track their
    steady states as a parameter varies.
    
    It is slightly more complicated to use continuation on a PDE. The below code demonstrates how.
    """
    """
    The PDE being used as an example is the 1D heat equation with homogenous Dirichlet boundary conditions
    u(0,t) = u(L,t) = 0  
    """

    """
    A function, f(u,args) must be defined which takes the PDE parameters as its args.
    It then passes them to solve_diffusive_pde() with the specified solving method and boundary type.
    The boundary condition and initial condition functions are defined with the function.
    """


    def pde_func(u, args):  # args [kappa, L, T , mx , mt]
        """
        The functions describing the boundary conditions and initial condition must be defined.
        """
        """
        boundary conditions u(0,t) = 0, u(L,t) = 0
        """

        def l_boundary(x, t):
            return 0

        def r_boundary(x, t):
            return 0

        """
        initial condition
        """

        def initial(x, t, L):
            # initial temperature distribution
            y = np.sin(pi * x / L)
            return y

        kappa = args[0]
        L = args[1]
        T = args[2]
        mx = args[3]
        mt = args[4]

        # values passed to solve_diffusive_pde()
        x, u_j = solve_diffusive_pde('crank', kappa, L, T, mx, mt, 'dirichlet', l_boundary, r_boundary, initial,
                                     ic_args=L)
        # PDE solution returned
        return u_j


    """
    a solver must be defined in the form f(f,u,args) which takes pde_func and its args and simply returns the function
    """


    def pde_solve(f, u, args):
        return f(u, args)


    """
    pde numerical values
    """
    kappa = 1
    L = 2
    T = 0.5
    mx = 100
    mt = 1000

    # values put in an array to pass to pde_func()
    args = np.array([kappa, L, T, mx, mt])

    """
    The above function are then passed to continuation()
    u0 is simply np.ones(mx+1) - value doesnt matter, just needs to be right size.
    discretisation = lambda x:x
    the parameter to vary, the range to vary it over, and how many intervals it is split into are selected as normal.
    In this case T is being varied from 0.5 to 5.
    """
    par_list, u = continuation('natural', pde_func, np.ones(mx + 1), args, 0, [0.5, 5], 10, lambda x: x, pde_solve)

    # get x values to plot u against
    x = np.linspace(0, L, mx + 1)

    # u needs to be transposed before plotting to make matplotlib happy
    plt.plot(x, np.transpose(u))
    plt.xlabel('x')
    plt.ylabel(f'u(x,T)')
    # generating legend labels from par_list
    labels = [f"T = {T}" for T in par_list]
    plt.legend(labels)
    plt.show()

    """
    The figures shows the solutions of the PDE plotted for different values of T. We can see that as T increases the
    PDE approaches the steady state. In this case the steady state can be seen to be u(x, inf) = 0.
    """

    """
    Example 5: More numerical continuation on PDEs.
    """
    """
    The PDE from Example 2 of solve_pde.main() is being used. It has Neumann boundary conditions and a source term.
    """


    def pde_func(u, args):  # args [kappa, L, T , mx , mt]
        """
        The functions describing the boundary conditions and initial condition must be defined.
        """
        """
        boundary conditions du/dx(0, t) = t, du/dx(L, t) = 1
        """

        def l_boundary(x, t):
            return 0

        def r_boundary(x, t):
            return 1

        """
        initial condition
        """

        def initial(x, t, L):
            # initial temperature distribution
            y = np.sin(pi * x / L)
            return y

        """
        source term F(x,t) = x + t
        """

        def source(x, t):
            return x + t

        kappa = args[0]
        L = args[1]
        T = args[2]
        mx = args[3]
        mt = args[4]

        # values passed to solve_diffusive_pde()
        x, u_j = solve_diffusive_pde('crank', kappa, L, T, mx, mt, 'neumann', l_boundary, r_boundary, initial, source,
                                     ic_args=L)
        # PDE solution returned
        return u_j


    """
    pde numerical values
    """
    kappa = 1
    L = 2
    T = 0.5
    mx = 100
    mt = 1000

    # values put in an array to pass to pde_func()
    args = np.array([kappa, L, T, mx, mt])

    """
    performing numerical continuation as in Example 4, and plotting the solutions.
    The solver is the pde_solve() function defined in Example 4.
    """
    par_list, u = continuation('natural', pde_func, np.ones(mx + 1), args, kappa, [1, 10], 4, lambda x: x, pde_solve)

    # get x values to plot u against
    x = np.linspace(0, L, mx + 1)

    # u needs to be transposed before plotting to make matplotlib happy
    plt.plot(x, np.transpose(u))
    plt.xlabel('x')
    plt.ylabel(f'u(x,0.5)')
    # generating legend labels from par_list
    labels = [f"kappa = {kappa}" for kappa in par_list]
    plt.legend(labels)
    plt.show()

    """
    The figure shows the PDE solutions for 4 different values of kappa.
    """
