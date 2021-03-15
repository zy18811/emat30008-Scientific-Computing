from solve_ode import solve_ode
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve



def func(t,y):
    x = y[0]
    y = y[1]
    a = 1
    d = 0.1
    b = 0.2
    dxdt = x*(1-x) - (a*x*y)/(d+x)
    dydt = b*y*(1-(y/x))
    return np.array([dxdt,dydt])



def repLineVal(vals,dp):
    round = np.around(vals, dp)
    unique, counts = np.unique(round, return_counts=True)
    val = unique[np.where(counts == np.max(counts))]
    return val[0]


def periodID(t,vals,dp=5):
    val = repLineVal(vals,dp)
    vals = np.around(vals,dp)
    valTs = t[np.where(vals == val)]
    Tarr = []
    for i in range(len(valTs)-1):
        Tarr.append(valTs[i+1]-valTs[i])
    T = np.min(Tarr)
    return val,T


def G(x,*args): # x = [u0,T], args = (f,phase)
    u01 = x[0]
    u02 = x[1]
    T = x[2]
    u0 = [u01,u02]
    f = args[0]
    phase = args[1]

    sol = f(u0,T)

    g = [u0[0]-sol[0],u0[1]-sol[1]]

    p = phase(u0)
    return np.array([g[0],g[1],p])



def f(u0,T):
    tArr = np.linspace(0,T,100)
    sol = solve_ode(func, u0, tArr, "rk4",0.01,system=True)
    return np.array([sol[0][-1],sol[1][-1]])


def pc(u0):
    x = u0[0]
    y = u0[1]
    a = 1
    d = 0.1
    b = 0.2
    p = x*(1-x) - (a*x*y)/(d+x)
    return p


'''
t = np.linspace(0,1000,10000)
eulsol = solve_ode(func,[0.25,0.25],t,"rk4",0.001,system=True)
xeul = eulsol[0]
yeul = eulsol[1]


ddt = func(t,[xeul,yeul])
dxdt = ddt[0]
dydt = ddt[1]


print(periodID(t,yeul,dp = 6))
#print(periodID(t,xeul))


plt.plot(t,yeul,label = "x")
plt.plot(t,yeul,label = "y")
#plt.legend()
#plt.plot(dxdt,dydt)
plt.show()
'''


t = np.linspace(0,1000,10000)
eulsol = solve_ode(func,np.array([0.25,0.25]),t,"rk4",0.001,system=True)
xeul = eulsol[0]
yeul = eulsol[1]

#print(fsolve(G,np.array([1,1,10]),args = (f,pc)))
x_period = periodID(t,xeul,dp = 5)
xval = x_period[0]
T = x_period[1]

xval_t_i  = np.where(np.around(xeul,5)==xval)[0][0]

xval_t = t[xval_t_i]
yval = yeul[np.where(t == xval_t)[0]]

t_step = 1000/10000

num_t_i = np.floor(T/t_step)

xval_t_i_next = int(xval_t_i+num_t_i)



xval_next = xeul[xval_t_i_next]
yval_next = yeul[xval_t_i_next]

print(xval)
print(yval)
print(xval_next)
print(yval_next)




plt.plot(xeul,yeul,label = "x")
plt.plot(xval,yval,'r+')
#plt.plot(xval_next,yval_next,'bx')
plt.show()