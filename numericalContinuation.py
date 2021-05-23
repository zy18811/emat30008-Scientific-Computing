from solve_ode import solve_ode
from shooting import orbitShooting,shootingG
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import fsolve,root
from newtonrhapson import newton


def continuation(method,function, u0, pars, vary_par, vary_par_range,vary_par_number = 100, discretisation = shootingG,solver = fsolve,pc = None):
    if method == 'natural':
        parArr = np.linspace(vary_par_range[0],vary_par_range[1],vary_par_number)
        def solWrapper(val,pc,u0):
            pars[vary_par] = val
            if pc is None:
                args=(pars)
            else:
                args=(pc,pars)
            return np.array(solver(discretisation(function), u0, args=args))
        sols = []
        for val in tqdm(parArr):
            u0 = solWrapper(val,pc,np.round(u0,2))
            sols.append(u0)
        sols = np.array(sols)
        Xs = sols[:, 0]
        Ys = sols[:,1]
        Ts = sols[:,2]
        plt.plot(parArr, Xs)
        plt.show()
    elif method == 'pseudo':
        pass


def hopfNormal(t,u,args):
    beta = args[0]
    sigma = args[1]
    sigma = -1
    u1 = u[0]
    u2 = u[1]
    du1dt = beta*u1 - u2 + sigma*u1*(u1**2+u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2+u2**2)
    return np.array([du1dt,du2dt])


def pcHopfNormal(u0,args):
    p = hopfNormal(1,u0,args)[0]
    return p

def cubic(x,c):
    return x**3 - x + c





u0_hopfNormal = np.array([1.4,0,6.3])

#continuation('natural',hopfNormal,u0_hopfNormal,[2, -1],0,[2,0],30,shootingG,fsolve,pcHopfNormal)
#continuation('natural',cubic,[1,1,1],[-2],0,[-2,2],30,discretisation= lambda x:x,solver=fsolve,pc=None)

p0 = 1.9
p1 = 2

true0 = orbitShooting(hopfNormal,u0_hopfNormal,pcHopfNormal,fsolve,[p0,-1])

true1 = orbitShooting(hopfNormal,np.round(true0,2),pcHopfNormal,fsolve,[p1,-1])

x0 = true0[:-1]
x1 = true1[:-1]
delta_x = x1 - x0
delta_p = p1-p0


def pseudo(x,p,delta_x,delta_p):
    pred_x = x+delta_x
    pred_p = p+delta_p
    arc = np.dot(delta_x,x-pred_x)+np.dot(delta_p,p-pred_p)
    return arc


def newG(x,ode,pc,delta_x,delta_p,*args):
    par = x[-1]
    T = x[-2]
    u0 = x[:-2]

    def F(u0, T):
       tArr = np.linspace(0, T, 1000)
       sol = solve_ode(ode, u0, tArr, "rk4", 0.01, True, *args)
       return sol[:,-1]
    g = np.append(u0-F(u0,T),[pc(u0,*args),pseudo(u0,par,delta_x,delta_p)])
    return g


p = 1.9
u0 = np.array([1.4,0,6.3,p])
sol = fsolve(newG,u0,args = (hopfNormal,pcHopfNormal,delta_x,delta_p,[p,-1]))
print(sol)









