from solve_ode import solve_ode
from shooting import orbitShooting,shootingG
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import fsolve,root
from newtonrhapson import newton
import sys


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

        normed_x = np.linalg.norm(np.stack((sols[:,0],sols[:,1]),axis = 1),axis = 1 )

        plt.plot(parArr,normed_x)
        #plt.show()

    elif method == 'pseudo':
        pass


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

def cubic(x,c):
    return x**3 - x + c

def pseudo(x,delta_x,p,delta_p):
    x_pred = x+delta_x
    p_pred = p+delta_p

    ds = np.linalg.norm(np.append(delta_x,delta_p))

    arc = np.dot(x-x_pred,delta_x) + np.dot(p-p_pred,delta_p) - ds
    return arc


def F(x,ode, pc, delta_x, delta_p,pars,vary_par):
    u0 = x[:-1] #+ delta_x
    p = x[-1] #+ delta_p
    pars[vary_par] = p
    G = shootingG(ode)
    g = G(u0,pc,pars)
    arc = pseudo(u0,delta_x,p,delta_p)
    ret = np.append(g,arc)
    return ret


u0_hopfNormal = np.array([1.4,0,6.3])

continuation('natural',hopfNormal,u0_hopfNormal,[2, -1],0,[2,-1],15,shootingG,fsolve,pcHopfNormal)
#continuation('natural',cubic,[1,1,1],[-2],0,[-2,2],30,discretisation= lambda x:x,solver=fsolve,pc=None)

ds = -0.2

p0 = 2
p1 = p0+ds

true0 = orbitShooting(hopfNormal,u0_hopfNormal,pcHopfNormal,fsolve,[p0,-1])

true1 = orbitShooting(hopfNormal,np.round(true0,2),pcHopfNormal,fsolve,[p1,-1])

#plt.plot(p2,x2[0],'k1')





x0 = true0
x1 = true1

delta_x = x1-x0
l_x = np.linalg.norm(delta_x)
delta_p = p1-p0
l_p = np.linalg.norm(delta_p)

u0 = np.append(x0,p0)
u1 = np.append(x1,p1)

plt.plot(p0,x0[0],'r+')
plt.plot(p1,x1[0],'r+')

#p2 = p1+delta_p
#x2 = x1+delta_x
#u2 = np.append(x2,p2)

point1 = u0
point2 = u1

for i in tqdm(range(8)):

    delta_x = point2[:-1]-point1[:-1]
    #delta_x *= l_x/np.linalg.norm(delta_x)
    delta_p = point2[-1]-point1[-1]
    #delta_p *= l_p/np.linalg.norm(delta_p)

    point_pred_x = point2[:-1] + delta_x
    point_pred_p = point2[-1] + delta_p

    plt.axline((point1[-1], point1[0]), (point2[-1], point2[0]), c='k', ls='dotted',alpha = 0.3)

    point_pred = np.append(point_pred_x,point_pred_p)

    u = point_pred

    plt.plot(u[-1],u[0],'g*')

    solution = root(F,u,method = 'lm',args = (hopfNormal,pcHopfNormal,delta_x,delta_p,[u[-1],-1],0))
    sol = solution['x']
    print(solution['fun'])

    plt.plot(sol[-1],sol[0],'gx')

    plt.axline((u[-1],u[0]),(sol[-1],sol[0]),c = 'k',ls = '--',alpha = 0.4)

    point1 = point2
    point2 = sol



plt.show()





