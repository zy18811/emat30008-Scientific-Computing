from shooting import orbitShooting
import numpy as np
import matplotlib.pyplot as plt

def func1(t,u):
    beta = 1
    sigma = -1
    u1 = u[0]
    u2 = u[1]
    du1dt = beta*u1 - u2 + sigma*u1*(u1**2+u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2+u2**2)
    return np.array([du1dt,du2dt])


def explicit(beta,phase,t):
    u1 = np.sqrt(beta) * np.cos(t+phase)
    u2 = np.sqrt(beta) * np.sin(t + phase)
    return u1,u2


def phaseCond(u0):
    u1 = u0[0]
    u2 = u0[1]

    beta = 1
    sigma = -1
    p = beta*u1 - u2 + sigma*u1*(u1**2+u2**2)
    return p


def main():
    t = np.linspace(0,100,1000)
    u0 = np.array([2,2,10])
    shoot = orbitShooting(func1,u0,phaseCond)
    x_shoot = shoot[0]
    y_shoot = shoot[1]
    sol_shoot = np.array([x_shoot,y_shoot])
    print(sol_shoot)
    T = shoot[2]
    sol = explicit(1,T,t)
    x = sol[0]
    y = sol[1]
    exp = explicit(1,T,0)
    x_exp = exp[0]
    y_exp = exp[1]
    sol_exp = np.array([x_exp,y_exp])
    print(sol_exp)
    print(np.allclose(sol_shoot,sol_exp))
    plt.plot(x,y)
    plt.plot(x_shoot,y_shoot,'r+',label = "shoot")
    plt.plot(x_exp,y_exp,'gx',label = "explicit")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()