import numpy as np


def approxJ(f,x,pc,dx = 1e-8,*args):
    n = len(x)
    func = f(x,pc,*args)
    jac = np.zeros((n,n))
    for j in range(n):
        Dxj = (abs(x[j])*dx) if x[j] != 0 else dx
        x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(x)]
        jac[:, j] = (f(x_plus,pc,*args) - func)/Dxj
    return jac


def newtonIter(f,x0,pc,*args):
    J = approxJ(f,x0,pc,1e-8,*args)
    #print(J)
    #invJ = np.linalg.inv(J)
    #x1 = x0 - np.matmul(invJ,f(x0,pc))
    x1_minus_x0 = np.linalg.solve(J,-f(x0,pc,*args))
    x1 = x1_minus_x0+x0
    return x1


def newton(f,x0,args):
    pc = args[0]
    args = args[1]
    while True:
        check = f(x0, pc, args)
        zero = np.zeros(np.shape(check))
        if np.allclose(check, zero):
            break
        x0 = newtonIter(f,x0,pc,args)
    return x0



