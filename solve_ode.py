import matplotlib.pyplot as plt
import numpy as np

"""
Euler and 4th order Runge-Kutta method implemented to solve ODE initial value problems.
"""


def euler_step(f, x0, t0, h, *args):
    """
    Performs a single step of the Euler method for given function at (t1,x1) with step size h
    :param f: Function definition of ODE(s) in the form f(t,x,*args) which returns derivative value at (t,x)
    :param x0: x value to start step at
    :param t0: t value to start step at
    :param h: Step size
    :param args: Array containing any additional arguments to be passed to the function
    :return: Returns value of function after step as x1, t1
    """
    x1 = x0 + h * f(t0, x0, *args)
    t1 = t0 + h
    return x1, t1


def rk4_step(f, x0, t0, h, *args):
    """
    Performs a single step of the 4th order Runge-Kutta method for given function at (t1,x1) with step size h
    :param f: Function definition of ODE(s) in the form f(t,x,*args) which returns derivative value at (t,x)
    :param x0: x value to start step at
    :param t0: t value to start step at
    :param h: Step size
    :param args: Array containing any additional arguments to be passed to the function
    :return: returns value of function after step as x1, t1
    """
    m1 = f(t0, x0, *args)
    m2 = f(t0 + h / 2, x0 + (h / 2) * m1, *args)
    m3 = f(t0 + h / 2, x0 + (h / 2) * m2, *args)
    m4 = f(t0 + h, x0 + h * m3, *args)
    x1 = x0 + (h / 6) * (m1 + 2 * m2 + 2 * m3 + m4)
    t1 = t0 + h
    return x1, t1


def solve_to(step, f, x1, t1, t2, deltat_max, *args):
    """
    Solves ODE from (t1,x1) to (t2,x2) with given values of t1,x1,t2 with a step size no larger than deltat_max
    :param step: step to use - euler_step and rk4_step implemented
    :param f: Function defining ODE(s) to solve in the form f(t,x,*args) which returns derivative value at (t,x)
    :param x1: x value of start point
    :param t1: t value of start point
    :param t2: t value of end point
    :param deltat_max: Maximum step size
    :param args: Array containing any additional arguments to be passed to the function
    :return: x value at t2 - ie. value of x2 for (t2,x2)
    """
    while (t1 + deltat_max) < t2:  # steps while t value is < t2
        x1, t1 = step(f, x1, t1, deltat_max, *args)
    else:
        x1, t1 = step(f, x1, t1, t2 - t1, *args)  # bridges gap between last step and t2 using step size t2-t_prev_step
    return x1


def integer_float_array_input_check(param_name, param):
    """
    Function to check type of x0, t_arr, and delat_max values. Raises a TypeError if not integer of float
    :param param_name: Name of parameter to check
    :param param: Parameter to check
    """
    not_int_bool = np.array(param).dtype != np.int_
    not_float_bool = np.array(param).dtype != np.float_
    if not_int_bool and not_float_bool:
        raise TypeError(f"{param_name}: '{param}' contains invalid types. {param_name} "
                        f"should contain integers and floats only.")


def solve_ode(f, x0, t_arr, method, deltat_max, system=False, *args):
    """
    Returns solution of an ODE or system of ODEs for an array of time values
    :param f: Function defining ODE(s) to solve in the form f(t,x,*args) which returns derivative value at (t,x)
    :param x0: Initial condition(s) of ODE(s) - integer or float for single ODE, ndarray for system of ODEs
    :param t_arr: Array of time values to solve for
    :param method: Solving method to use - "euler" for Euler method, "rk4" for 4th order Runge-Kutta method
    :param deltat_max: Maximum step size to be used while solving
    :param system: Boolean for whether it is a system of ODEs to be solved - True: system, False: not a system
    :param args: Array containing additional args to be passed to the function
    :return: Returns an array containing the x value(s) for each time value in t_arr
    """
    """
    checks type(s) of x0
    """
    integer_float_array_input_check('x0', x0)

    """
    checks type of t_arr
    """
    integer_float_array_input_check('t_arr', t_arr)

    """
    checks type of deltat_max
    """
    integer_float_array_input_check('deltat_max', deltat_max)

    """
    checks if f is a function. If it is checks if it returns an output in the right shape. Raises an error if not
    """
    if callable(f):

        # tests that function output has same shape as x0
        t = t_arr[0]
        test = f(t, x0, *args)
        if isinstance(test, (int, np.int_, np.float_, list, np.ndarray)):
            if not np.array(test).shape == np.array(x0).shape:
                raise ValueError("Shape mismatch. Shape of x0 and f output not the same")
        else:
            raise TypeError(f"Output of f is {type(test)}. Output needs to be of type int, float, list or ndarray")
    else:
        raise TypeError(f"f: '{f}' needs to be a function.")

    """
    checks if system is a bool, raises a TypeError if not
    """
    if not isinstance(system, bool):
        raise TypeError(f"system: '{system}' contains invalid types. system should be boolean only.")

    """"
    checks if method param is valid, raises ValueError if not
    """
    if method == "euler":
        step = euler_step
    elif method == "rk4":
        step = rk4_step
    else:
        raise ValueError(f"method: '{method}' is not valid. Please select 'euler' or 'rk4'.")

    """
    initialises solution_array, shape different depending 'system' bool
    """
    if system:
        solution_array = np.empty(shape=(len(t_arr), len(x0)))
    else:
        solution_array = np.empty(shape=(len(t_arr), 1))
    solution_array[0] = x0

    """
    iterates through t_arr applying solve_to on pairwise t values
    inserts x value(s) for each t value into solution_array
    """
    for i in range(len(t_arr) - 1):
        xi = solve_to(step, f, solution_array[i], t_arr[i], t_arr[i + 1], deltat_max, *args)
        solution_array[i + 1] = xi

    """
    returns solution_array
    output needs to be reshaped if system = True
    """
    if system:
        return solution_array.transpose()
    else:
        return solution_array


def main():
    """
    Example 1 - Single ODE

    dx/dt = x,  initial condition x(0) = 1

    Solving for t = 0 to t = 1
    """

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

    t = np.linspace(0, 1, 100)
    """
    Euler, h = 0.01
    """
    ex1_euler_sol = solve_ode(dxdt_equals_x, 1, t, 'euler', 0.01, False)

    """
    4th Order Runge-Kutta, h = 0.01
    """
    ex1_rk4_sol = solve_ode(dxdt_equals_x, 1, t, 'rk4', 0.01, False)

    """
    Plotting solutions and true solution
    """
    plt.plot(t, ex1_euler_sol, label='Euler')
    plt.plot(t, ex1_euler_sol, label='RK4')
    plt.plot(t, dxdt_equals_x_true(t), label='True')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    plt.show()

    """
    Example 2 - System of ODEs

    d2x/dt2 = -x,  initial condition x(0) = 1
    
    This is equivalent to the system of ODEs:
    
    dx/dt = y, dy/dt = -x, initial conditions x(0) = 1, y(0) = 1

    Solving for t = 0 to t = 10
    """

    def d2xdt2_equals_minus_x(t, u):
        """
        Function defining system of  ODEs dx/dt = y, dy/dt = -x
        :param t: t value
        :param u: vector u = [x, y]
        :return: returns value of dx/dt and dy/dt at (t,u)
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

    t = np.linspace(0, 10, 100)
    """
    Euler, h = 0.01
    """
    ex2_euler_sol = solve_ode(d2xdt2_equals_minus_x, [1, 1], t, 'rk4', 0.01, True)
    ex2_euler_sol_x = ex2_euler_sol[0]
    ex2_euler_sol_y = ex2_euler_sol[1]

    """
    4th Order Runge-Kutta, h = 0.01
    """
    ex2_rk4_sol = solve_ode(d2xdt2_equals_minus_x, [1, 1], t, 'rk4', 0.01, True)
    ex2_rk4_sol_x = ex2_rk4_sol[0]
    ex2_rk4_sol_y = ex2_rk4_sol[1]

    """
    Plotting solutions and true solution
    """
    true = d2xdt2_equals_minus_x_true(t)
    true_x = true[0]
    true_y = true[1]

    plt.subplot(2, 1, 1)
    plt.plot(t, ex2_euler_sol_x, label='Euler')
    plt.plot(t, ex2_rk4_sol_x, label='RK4')
    plt.plot(t, true_x, label='True')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('x')

    plt.subplot(2, 1, 2)
    plt.plot(t, ex2_euler_sol_y, label='Euler')
    plt.plot(t, ex2_rk4_sol_y, label='RK4')
    plt.plot(t, true_y, label='True')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('y (dx/dt)')
    plt.show()


if __name__ == '__main__':
    main()
