from solve_ode import solve_ode
import numpy as np
import matplotlib.pyplot as plt

def func(t,y):
    x = y[0]
    y = y[1]
    a = 1
    d = 0.1
    b = 0.1
    dxdt = x*(1-x) - (a*x*y)/(d+x)
    dydt = b*y*(1-(y/x))
    return np.array([dxdt,dydt])


t = np.linspace(0,500,1000)
eulsol = solve_ode(func,[0.25,0.25],t,"euler",0.01,system=True)
xeul = eulsol[0]
yeul = eulsol[1]



plt.plot(t,xeul)
plt.plot(t,yeul)
plt.show()