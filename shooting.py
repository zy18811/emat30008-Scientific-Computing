import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from periodfinderforcheck import manual_period_finder
from solve_ode import solve_ode, integer_float_input_check

"""
Functions implementing numerical shooting techniques to find the periodic orbits of ODEs/systems of ODEs.
"""


def orbit_shooting(ode, u0, pc, solver, *args):
    """
    Uses numerical shooting to locate the periodic orbit, if any, of a given ODE/system of ODEs.
    The orbit is defined by coordinates of its starting point and its time period.
    :param ode: Function defining ODE(s) to find orbit of in the form f(t,x,*args) which returns derivative at (t,x)
    :param u0: Numpy array of the initial guess for location of periodic orbit
    :param pc: Phase condition
    :param solver: Solver to be used - fsolve or newton. Fsolve performs better
    :param args: Array containing additional args to be passed to the function
    :return: Returns the start coordinates and time period of found orbit. If the root finding fails, returns an empty
    array.
    """
    """
    checks type of u0 - see solve_ode.py for function details
    """
    integer_float_input_check("u0", u0)

    """
    checks if ode is a function. If it is checks if it returns an output in the right shape. Raises an error if not
    """

    if callable(ode):

        # tests that function output is an int, float, list or ndarray
        with np.errstate(over='raise'):
            # if root finding fails for initial values, returns an empty array
            try:
                test = ode(0, u0, *args)
            except FloatingPointError:
                return []
        if isinstance(test, (int, np.int_, np.float_, list, np.ndarray)):
            # tests that function output has same shape as u0
            if not np.array(test).shape == np.array(u0[:-1]).shape:
                raise ValueError("Shape mismatch. Shape of u0 and ode output not the same")
        else:
            raise TypeError(f"Output of ode is {type(test)}. Output needs to be of type int, float, list or ndarray")
    else:
        raise TypeError(f"ode: '{ode}' needs to be a function.")

    """
    checks if pc is a function. If it is checks if it returns a scalar output. Raises an error if not
    """

    if callable(pc):

        # tests that phase condition output is an int or float
        test = pc(u0, *args)
        if not isinstance(test, (int, float, np.int_, np.float_)):
            raise TypeError(
                f"Output of phase condition is {type(test)}. Output needs to be of type int or float")
    else:
        raise TypeError(f"pc: '{pc}' needs to be a function.")

    G = shootingG(ode)  # Shooting root finding problem, G, for given ode

    with np.errstate(over='raise'):
        # if root finding fails returns an empty array
        try:
            orbit = solver(G, u0, args=(pc, *args))  # Finds root of G, yielding the location of any periodic orbit
        except FloatingPointError:
            return []
    return orbit


def shootingG(ode):
    """
    Constructs the shooting root finding problem for given ODE
    :param ode: Function defining ODE(s) to solve in the form f(t,x,*args) which returns derivative value at (t,x)
    :return: Returns the function, G,  whose root solves the shooting problem.
    """

    def G(x, pc, *args):
        """
        Vector function of (initial guess - solution, phase condition). The root of this function yields the periodic
        orbit if there is one.
        :param x: Array containing initial guess of coordinates and time period. The time period occupies the last value
        :param pc: Phase condition function
        :param args: Any additional args to be passed to function and phase condition
        :return:
        """

        def F(u0, T):
            """
            Returns solution of ODE estimated using 4th order Runge-Kutta method at time T and initial conditions u0.
            :param u0: Initial conditions for ODE
            :param T: Time T to solve at
            :return: Returns solution of ODE at time T
            """
            tArr = np.linspace(0, T, 1000)
            sol = solve_ode(ode, u0, tArr, "rk4", 0.01, True, *args)
            return sol[:, -1]

        T = x[-1]
        u0 = x[:-1]
        g = np.append(u0 - F(u0, T), pc(u0, *args))  # Constructs array of ((initial guess - solution, phase condition)
        return g

    return G


def main():
    """
    Function for predator-prey equations
    """

    def predator_prey(t, y, args):
        x = y[0]
        y = y[1]

        a = args[0]
        d = args[1]
        b = args[2]

        dxdt = x * (1 - x) - (a * x * y) / (d + x)
        dydt = b * y * (1 - (y / x))
        return np.array([dxdt, dydt])

    """
    Phase condition for predator-prey equations
    """

    def pc_predator_prey(u0, args):
        return predator_prey(0, u0, args)[0]

    """
    initial values guess for predator-prey equations
    """
    predator_prey_u0 = np.array([0.07, 0.16, 23])

    """
    b = 0.4 > 0.26 case
    """
    args = [1, 0.1, 0.4]
    """
    solving predator-prey equations with orbit coordinates as initial conditions for t = 0 to t = 100
    Using RK4 with h = 0.01
    """
    t = np.linspace(0, 1000, 1000)
    predator_prey_solution = solve_ode(predator_prey, predator_prey_u0[:-1], t, 'rk4', 0.01, True, args)
    predator_prey_solution_x = predator_prey_solution[0]
    predator_prey_solution_y = predator_prey_solution[1]

    """
    Plotting y against x 
    """
    plt.plot(predator_prey_solution_x, predator_prey_solution_y)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

    """
    b = 0.15 < 0.26 case
    """
    args = [1, 0.1, 0.15]

    """
    solving predator-prey equations with orbit coordinates as initial conditions for t = 0 to t = 100
    Using RK4 with h = 0.01
    """
    t = np.linspace(0, 1000, 1000)
    predator_prey_solution = solve_ode(predator_prey, predator_prey_u0[:-1], t, 'rk4', 0.01, True, args)
    predator_prey_solution_x = predator_prey_solution[0]
    predator_prey_solution_y = predator_prey_solution[1]
    """
    Plotting y against x 
    """
    plt.plot(predator_prey_solution_x, predator_prey_solution_y)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

    """
    It can be seen from the figures that the predator prey equations have different behaviours in the long term limit
    depending on the value of b. For b > 0.26, the values of x and y tend towards a steady state value. This is shown on
    the plot of x against y as an inwards spiral towards a point. For b < 0.26, the equations fall into a period orbit.
    This is shown in the plot of x against y as a circular shape.
    """
    """
    The orbit can be manually found by analysing the time series using manual_period_finder() from
    periodfinderforcheck.py
    """
    manual_orbit = manual_period_finder(t, predator_prey_solution_x, predator_prey_solution_y)
    """
    Plotting the location of this orbit on the graph of y against x
    """
    plt.plot(manual_orbit[0], manual_orbit[1], 'go', label="Manually found orbit")
    plt.plot(predator_prey_solution_x, predator_prey_solution_y)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend()
    plt.show()
    """
    Now trying to find the same orbit using numerical shooting by using the orbit_shooting() function
    """
    shooting_orbit = orbit_shooting(predator_prey, predator_prey_u0, pc_predator_prey, fsolve, args)
    """
    Plotting this orbit onto the figure as well.
    """
    plt.plot(shooting_orbit[0], shooting_orbit[1], 'ro', label="Numerical Shooting Orbit")
    plt.plot(manual_orbit[0], manual_orbit[1], 'go', label="Manually Found Orbit")
    plt.plot(predator_prey_solution_x, predator_prey_solution_y)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend()
    plt.show()

    """
    It can be seen from the figure that the two orbits are the same and that the numerical shooting was successful.
    """


if __name__ == "__main__":
    main()
