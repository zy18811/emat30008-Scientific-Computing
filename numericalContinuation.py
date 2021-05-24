from solve_ode import solve_ode
from shooting import orbitShooting,shootingG
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import fsolve,root
from newtonrhapson import newton
import sys


def continuation(method,function, u0, pars, vary_par, vary_par_range,vary_par_number = 100, discretisation = shootingG,solver = fsolve,pc = None):

    def solWrapper(val,u0):
        pars[vary_par] = val
        if pc is None:
            args = (pars)
        else:
            args = (pc, pars)
        return np.array(solver(discretisation(function), u0, args=args))

    if method == 'natural':
        par_list, x = natural_parameter_continuation(u0,vary_par_range,vary_par_number,solWrapper)
    elif method == 'pseudo':
        par_list, x = pseudo_arclength_continuation(function,u0,pars,vary_par,vary_par_range,vary_par_number,discretisation,pc,solWrapper)
    else:
        sys.exit("Method: \"%s\" is not valid. Please select 'natural' or 'pseudo'." % method)
    normed_x = np.linalg.norm(np.delete(x, -1, axis=1), axis=1)
    plt.plot(par_list,x[:,0])



def hopfNormal(t,u,args):
    beta = args[0]
    sigma = args[1]

    u1 = u[0]
    u2 = u[1]
    du1dt = beta*u1 - u2 + sigma*u1*(u1**2+u2**2)
    if du1dt == np.NaN:
        print(u1,u2,beta)

    du2dt = u1 + beta*u2 + sigma*u2*(u1**2+u2**2)
    return np.array([du1dt,du2dt])


def pcHopfNormal(u0,args):
    p = hopfNormal(1,u0,args)[0]
    return p


def modHopfNormal(t,u,args):
    beta = args[0]

    u1 = u[0]
    u2 = u[1]

    du1dt = beta*u1 - u2 + u1 * (u1**2+u2**2) - u1 * (u1**2+u2**2)**2
    du2dt = u1 + beta*u2 + u2 * (u1**2+u2**2) - u2 * (u1**2+u2**2)**2
    return np.array([du1dt,du2dt])


def pcModHopfNormal(u0,args):
    p = modHopfNormal(1,u0,args)[0]
    return p


def cubic(x,c):
    return x**3 - x + c


def natural_parameter_continuation(u0,vary_par_range,vary_par_number,solWrapper):
    parArr = np.linspace(vary_par_range[0], vary_par_range[1], vary_par_number)

    sols = []
    for val in tqdm(parArr):
        u0 = solWrapper(val,u0)
        sols.append(u0)
        u0 = np.round(u0,2)
    sols = np.array(sols)
    return parArr,sols



def pseudo_arclength_continuation(function, u0,  pars, vary_par, vary_par_range,vary_par_number, discretisation,pc,solWrapper):


    def pseudoGetTwoTrue(u0, dp):
        dp *= np.sign(vary_par_range[1]-vary_par_range[0])
        p0 = vary_par_range[0]
        p1 = p0 + dp
        true0 = np.append(solWrapper(p0,u0),p0)
        true1 = np.append(solWrapper(p1,np.round(true0[:-1],2)),p1)
        return true0,true1

    def pseudo(x, delta_x, p, delta_p):
        x_pred = x + delta_x
        p_pred = p + delta_p
        ds = np.linalg.norm(np.append(delta_x, delta_p))
        arc = np.dot(x - x_pred, delta_x) + np.dot(p - p_pred, delta_p) - ds
        return arc

    def F(x, ode, pc, discretisation, delta_x, delta_p, pars, vary_par):
        u0 = x[:-1]  # + delta_x
        p = x[-1]  # + delta_p
        pars[vary_par] = p
        d = discretisation(ode)
        if pc is None:
            g = d(u0, pars)
        else:
            g = d(u0,pc,pars)
        arc = pseudo(u0, delta_x, p, delta_p)
        f = np.append(g, arc)
        return f

    v0,v1 = pseudoGetTwoTrue(u0,0.05)

    #plt.plot(v0[-1], v0[0], 'r+')
    #plt.plot(v1[-1], v1[0], 'r+')

    sols = []
    par_list = []

    while True:
        delta_x = v1[:-1] - v0[:-1]
        #delta_x *= l_x/np.linalg.norm(delta_x)
        delta_p = v1[-1] - v0[-1]
        #delta_p *= l_p/np.linalg.norm(delta_p)

        pred_v_x = v1[:-1] + delta_x
        pred_v_p = v1[-1] + delta_p

        # plt.axline((point1[-1], point1[0]), (point2[-1], point2[0]), c='k', ls='dotted',alpha = 0.3)

        pred_v = np.append(pred_v_x, pred_v_p)
        pars[vary_par] = pred_v[-1]
        # plt.plot(u[-1],u[0],'g*')

        solution = root(F, pred_v, method='lm', args=(function, pc,discretisation, delta_x, delta_p, pars, 0))
        sol = solution['x']

        if sol[0] < 0:
            break

        sols.append(sol[:-1])
        par_list.append(sol[-1])




        #plt.plot(sol[-1], sol[0], 'gx')

        # plt.axline((u[-1],u[0]),(sol[-1],sol[0]),c = 'k',ls = '--',alpha = 0.4)

        v0 = np.round(v1,2)
        v1 = np.round(sol,2)


    sols = np.array(sols)

    return par_list,sols



u0_hopfNormal = np.array([1.4,0,6.3])

#continuation('natural',hopfNormal,u0_hopfNormal,[2, -1],0,[2,-1],100,shootingG,fsolve,pcHopfNormal)
#continuation('pseudo',hopfNormal,u0_hopfNormal,[2,-1],0,[2,-1],200,shootingG,fsolve,pcHopfNormal)

continuation('natural',modHopfNormal,u0_hopfNormal,[2],0,[2,-1],50,shootingG,fsolve,pcModHopfNormal)
continuation('pseudo',modHopfNormal,u0_hopfNormal,[2],0,[2,-1],30,shootingG,fsolve,pcModHopfNormal)


#continuation('natural',cubic,[1,1,1],[-2],0,[-2,2],200,discretisation= lambda x:x,solver=fsolve,pc=None)
#continuation('pseudo',cubic,np.array([1,1,1]),[2],0,[-2,2],200,lambda x:x,fsolve,None)


plt.grid()
plt.show()



