import numpy as np
import sys
from scipy.linalg import solve

def approxJ(f,x,dx = 1e-8,*args):
    n = len(x)
    func = f(x,*args)
    jac = np.zeros((n,n))
    for j in range(n):
        Dxj = (abs(x[j])*dx) if x[j] != 0 else dx
        x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(x)]
        jac[:, j] = (f(x_plus,*args) - func)/Dxj
    return jac


def newtonIter(f,x0,*args):
    J = approxJ(f,x0,1e-8,*args)
    try:
        x1_minus_x0 = solve(J,-f(x0,*args))
    except np.linalg.LinAlgError:
        sys.exit(f"Singular Jacobian --> Initial guess has caused solution to diverge.\n"
                 "Please try again with a different initial guess.")
    x1 = x1_minus_x0+x0
    return x1


def newton(f,x0,args):

    while True:
        check = f(x0,*args)
        zero = np.zeros(np.shape(check))
        if np.allclose(check, zero):
            break
        x0 = newtonIter(f,x0,*args)
    return x0



