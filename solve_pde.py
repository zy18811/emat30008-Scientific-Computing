import numpy as np
import pylab as pl
from math import pi
import scipy.sparse
import scipy.sparse.linalg


def tri_diag(size, diag1, diag2, diag3):
    diagonals = np.array([diag1 * np.ones(size-1), diag2 *  np.ones(size), diag3 * np.ones(size - 1)], dtype=object)
    offset = [-1, 0, 1]
    return scipy.sparse.diags(diagonals, offset, format='csr')


def forward_euler(kappa, L, T, mx, mt):
    def u_I(x):
        # initial temperature distribution
        y = np.sin(pi * x / L)
        return y

    x = np.linspace(0, L, mx + 1)  # mesh points in space
    t = np.linspace(0, T, mt + 1)  # mesh points in time
    deltax = x[1] - x[0]  # gridspacing in x
    deltat = t[1] - t[0]  # gridspacing in t
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number

    u_j = np.zeros(x.size)  # u at current time step
    u_jp1 = np.zeros(x.size)  # u at next time step

    A_FE = tri_diag(mx-1,lmbda,1-2*lmbda,lmbda)

    u_j_inner = u_I(x[1:mx])

    for j in range(0,mt):
        u_j_inner *= A_FE

    u_j = np.zeros(x.size)
    u_j[1:mx] = u_j_inner

    def u_exact(x, t):
        # the exact solution
        y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
        return y

    pl.plot(x, u_j, 'ro', label='forward')
    xx = np.linspace(0, L, 250)
    pl.plot(xx, u_exact(xx, T), 'b-', label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    #pl.show()


def backwards_euler(kappa,L,T,mx,mt):
    def u_I(x):
        # initial temperature distribution
        y = np.sin(pi * x / L)
        return y

    x = np.linspace(0, L, mx + 1)  # mesh points in space
    t = np.linspace(0, T, mt + 1)  # mesh points in time
    deltax = x[1] - x[0]  # gridspacing in x
    deltat = t[1] - t[0]  # gridspacing in t
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number

    A_BE = tri_diag(mx-1,-lmbda,1+2*lmbda,-lmbda)

    u_j_inner = u_I(x[1:mx])
    for j in range(0,mt):
        u_j_inner = scipy.sparse.linalg.spsolve(A_BE,u_j_inner)

    u_j = np.zeros(x.size)
    u_j[1:mx] = u_j_inner

    def u_exact(x, t):
        # the exact solution
        y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
        return y

    pl.plot(x, u_j, 'go', label='backward')
    xx = np.linspace(0, L, 250)
    pl.plot(xx, u_exact(xx, T), 'b-', label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')



def crank_nicholson(kappa,L,T,mx,mt):
    def u_I(x):
        # initial temperature distribution
        y = np.sin(pi * x / L)
        return y

    x = np.linspace(0, L, mx + 1)  # mesh points in space
    t = np.linspace(0, T, mt + 1)  # mesh points in time
    deltax = x[1] - x[0]  # gridspacing in x
    deltat = t[1] - t[0]  # gridspacing in t
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number

    A_CN = tri_diag(mx-1,-lmbda/2,1+lmbda,-lmbda/2)
    B_CN = tri_diag(mx-1,lmbda/2,1-lmbda,lmbda/2)

    u_j_inner = u_I(x[1:mx])
    for j in range(0,mt):
        u_j_inner = scipy.sparse.linalg.spsolve(A_CN,B_CN*u_j_inner)

    u_j = np.zeros(x.size)
    u_j[1:mx] = u_j_inner

    def u_exact(x, t):
        # the exact solution
        y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
        return y

    pl.plot(x, u_j, 'yo', label='crank')
    xx = np.linspace(0, L, 250)
    pl.plot(xx, u_exact(xx, T), 'b-', label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()


forward_euler(1,1,0.5,10,1000)
backwards_euler(1,1,0.5,10,1000)
crank_nicholson(1,1,0.5,10,1000)

