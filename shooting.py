from newtonrhapson import newton
from solve_ode import solve_ode
import numpy as np
from scipy.optimize import fsolve



def orbitShooting(ode,u0,pc,solver = fsolve,*args):
    G = shootingG(ode)
    orbit = solver(G, u0, args=(pc,*args))
    return orbit


def shootingG(ode):
    def G(x,pc,*args):
        def F(u0, T):
            tArr = np.linspace(0, T, 1000)
            sol = solve_ode(ode, u0, tArr, "rk4", 0.01, True, *args)
            return sol[:,-1]
        T = x[-1]
        u0 = x[:-1]
        g = np.append(u0 - F(u0, T), pc(u0, *args))
        return g
    return G


def pc(u0):
    print("hi")
    x = u0[0]
    y = u0[1]
    a = 1
    d = 0.1
    b = 0.1
    p = x*(1-x) - (a*x*y)/(d+x)
    return p


def func(t,y):
    x = y[0]
    y = y[1]
    a = 1
    d = 0.1
    b = 0.16
    dxdt = x*(1-x) - (a*x*y)/(d+x)
    dydt = b*y*(1-(y/x))
    return np.array([dxdt,dydt])


def main():
    x0 = np.array([0.5,0.5,15])
    print(orbitShooting(func,x0,pc))


if __name__ == "__main__":
    main()





