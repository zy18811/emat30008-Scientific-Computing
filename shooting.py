from newtonrhapson import newton
from solve_ode import solve_ode
import numpy as np


def orbitShooting(ode,u0,pc):
    def F(u0,T):
        tArr = np.linspace(0, T, 1000)
        sol = solve_ode(ode, u0, tArr, "rk4", 0.01, system=True)
        return np.array([sol[0][-1], sol[1][-1]])

    def G(x,pc):
        k = len(x)-1
        T = x[-1]
        u0 = x[0:k]
        sol = F(u0,T)
        g = np.empty(np.shape(u0))
        for i in range(np.shape(u0)[0]):
            g[i] = u0[i] - sol[i]
        p = pc(u0)
        ret = [g[i] for i in range(len(g))]
        ret.append(p)
        return np.array(ret)
    newt = newton(G,u0,pc)
    return newt


def pc(u0):
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

x0 = np.array([0.5,0.5,15])
print(orbitShooting(func,x0,pc))






