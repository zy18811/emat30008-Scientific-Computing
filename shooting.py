from newtonrhapson import newton
from solve_ode import solve_ode
import numpy as np
from scipy.optimize import fsolve


"""
Functions implementing numerical shooting techniques to find the periodic orbits of ODEs/systems of ODEs.
"""


def orbitShooting(ode,u0,pc,solver = fsolve,*args):
    """
    Uses numerical shooting to locate the periodic orbit, if any, of a given ODE/system of ODEs.
    The orbit is defined by coordinates of its starting point and its time period.
    :param ode: Function defining ODE(s) to solve in the form f(t,x,*args) which returns derivative value at (t,x)
    :param u0: Array of the initial guess for location of periodic orbit
    :param pc: Phase condition
    :param solver: Solver to be used - fsolve or newton. Fsolve performs better
    :param args: Array containing additional args to be passed to the function
    :return: Returns the start coordinates and time period of found orbit
    """
    G = shootingG(ode) # Shooting root finding problem, G, for given ode
    orbit = solver(G, u0, args=(pc,*args)) # Finds root of G, yielding the location of any periodic orbit
    return orbit


def shootingG(ode):
    """
    Constructs the shooting root finding problem for given ODE
    :param ode: Function defining ODE(s) to solve in the form f(t,x,*args) which returns derivative value at (t,x)
    :return: Returns the function, G,  whose root solves the shooting problem.
    """
    def G(x,pc,*args):
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
            return sol[:,-1]
        T = x[-1]
        u0 = x[:-1]
        g = np.append(u0 - F(u0, T), pc(u0, *args)) # Constructs array of ((initial guess - solution, phase condition)
        return g
    return G




def main():

    def pc(u0):
        x = u0[0]
        y = u0[1]
        a = 1
        d = 0.1
        b = 0.1
        p = x * (1 - x) - (a * x * y) / (d + x)
        return p

    def func(t, y):
        x = y[0]
        y = y[1]
        a = 1
        d = 0.1
        b = 0.16
        dxdt = x * (1 - x) - (a * x * y) / (d + x)
        dydt = b * y * (1 - (y / x))
        return np.array([dxdt, dydt])

    x0 = np.array([0.5,0.5,15])
    print(orbitShooting(func,x0,pc,solver=newton))


if __name__ == "__main__":
    main()





