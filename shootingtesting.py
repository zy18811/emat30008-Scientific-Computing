from shooting import orbitShooting
import numpy as np
import matplotlib.pyplot as plt


def shootingValueTest(func,u0,pc,explicit,testlabel = ""):

    xShoot,yShoot,T = orbitShooting(func,u0,pc)
    xTrue,yTrue = explicit(1,T,T)

    shoot = [xShoot, yShoot]
    true = [xTrue, yTrue]
    if np.allclose(shoot,true):
        print(f"{testlabel+':' if testlabel != '' else ''} Successful")
    else:
        print(f"{testlabel+':' if testlabel != '' else ''} Failed")

    t = np.linspace(0, 100, 1000)
    xSol,ySol = explicit(1,T,t)


    plt.plot(xSol,ySol)
    plt.plot(xShoot,yShoot,'r+',label = "Shoot")
    plt.plot(xTrue,yTrue,'bx',label = "True")
    plt.legend()
    plt.show()



def hopfNormal(t,u):
    beta = 1
    sigma = -1
    u1 = u[0]
    u2 = u[1]

    du1dt = beta*u1 - u2 + sigma*u1*(u1**2+u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2+u2**2)
    return np.array([du1dt,du2dt])


def hopfNormalExplicit(beta,phase,t):
    u1 = np.sqrt(beta) * np.cos(t+phase)
    u2 = np.sqrt(beta) * np.sin(t + phase)
    return u1,u2


def pcHopfNormal(u0):
    p = hopfNormal(1,u0)[0]
    return p


def test2(t,u):
    x = u[0]
    y = u[1]
    dxdt = y
    dydt = -x
    return np.array([dxdt,dydt])


def pc_test2(u0):
    return u0[0] - 1


def test2_explicit(beta,phase,t):
    x = np.sin(t) + np.cos(t)
    y = np.cos(t) - np.sin(t)
    return x,y



def main():
    u0_hopfNormal = np.array([0.9,0.1,6])
    shootingValueTest(hopfNormal,u0_hopfNormal,pcHopfNormal,hopfNormalExplicit,testlabel="Hopf bifurcation normal form")

    u0_test2 = np.array([1,1,2*np.pi])
    shootingValueTest(test2,u0_test2,pc_test2,test2_explicit,testlabel="Test 2")


if __name__ == "__main__":
    main()