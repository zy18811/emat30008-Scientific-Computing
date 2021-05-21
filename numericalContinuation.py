from solve_ode import solve_ode
from shooting import orbitShooting
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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


betaValues = np.linspace(0,2,1000)

u0_hopfNormal = np.array([0.9,0.1,6])

sols = []
for val in tqdm(betaValues):
    beta = val
    xshoot, yshoot, T = orbitShooting(hopfNormal,u0_hopfNormal,pcHopfNormal,[beta,-1])
    u0_hopfNormal = np.array([xshoot,yshoot,T])
    sols.append(u0_hopfNormal)
    #print(f"Done Beta = {val}")


sols = np.array(sols)
Xs = [item[0] for item in sols]
Ys = [item[1] for item in sols]
Ts = [item[2] for item in sols]

plt.plot(betaValues,Ts)
plt.show()

plt.plot(betaValues,Xs)
plt.show()

plt.plot(betaValues,Ys)
plt.show()
