import numpy as np


def approxJ(f,x,pc,dx = 1e-8):
    n = len(x)
    func = f(x,pc)
    jac = np.zeros((n,n))
    for j in range(n):
        Dxj = (abs(x[j])*dx) if x[j] != 0 else dx
        x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(x)]
        jac[:, j] = (f(x_plus,pc) - func)/Dxj
    return jac


def newtonIter(f,x0,pc):

    J = approxJ(f,x0,pc)
    #print(J)
    #invJ = np.linalg.inv(J)
    #x1 = x0 - np.matmul(invJ,f(x0,pc))
    x1_minus_x0 = np.linalg.solve(J,-f(x0,pc))
    x1 = x1_minus_x0+x0
    return x1


def newton(f,x0,pc):
    while True:
        check = f(x0, pc)
        zero = np.zeros(np.shape(check))
        if np.allclose(check, zero):
            break
        x0 = newtonIter(f,x0,pc)
    return x0



