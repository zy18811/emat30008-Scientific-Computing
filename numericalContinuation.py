from solve_ode import solve_ode
from shooting import orbitShooting,shootingG
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import fsolve
from newtonrhapson import newton


def nat_param_continuation(function, u0, pars, vary_par, vary_par_range,vary_par_number = 100, discretisation = shootingG,solver = fsolve,pc = None):

    parArr = np.linspace(vary_par_range[0],vary_par_range[1],vary_par_number)

    def solWrapper(val,pc):
        pars[vary_par] = val
        if pc is None:
            return np.array(solver(discretisation(function), u0, args=(pars)))
        else:
            return np.array(solver(discretisation(function), u0, args=(pc,pars)))

    sols = np.array([solWrapper(val,pc) for val in tqdm(parArr)])
    Xs = sols[:, 0]
    plt.plot(parArr, Xs)
    plt.show()


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


betaValues = np.linspace(2,0,30)

u0_hopfNormal = np.array([0.9,0.1,6])

def function(x,c):
    return x**3 - x + c


nat_param_continuation(hopfNormal,u0_hopfNormal,[2, -1],0,[2,0],30,shootingG,fsolve,pcHopfNormal)
nat_param_continuation(function,[1,1,1],[-2],0,[-2,2],30,discretisation= lambda x:x,solver=fsolve,pc=None)
