import numpy as np
import pylab as pl
from math import pi
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import warnings


def solve_diffusive_pde(method,kappa, L, T, mx, mt, boundary_type,l_boundary_condition,r_boundary_condition,
                        initial_condition,source = None):

    def tri_diag(size, diag1, diag2, diag3):
        diagonals = np.array([diag1 * np.ones(size - 1), diag2 * np.ones(size), diag3 * np.ones(size - 1)],
                             dtype=object)
        offset = [-1, 0, 1]
        return scipy.sparse.diags(diagonals, offset, format='csc')


    def identity(size):
        return tri_diag(size,0,1,0)

    warnings.filterwarnings('ignore',category=scipy.sparse.SparseEfficiencyWarning)
    x = np.linspace(0, L, mx + 1)  # mesh points in space
    t = np.linspace(0, T, mt + 1)  # mesh points in time

    deltax = x[1] - x[0]  # gridspacing in x
    deltat = t[1] - t[0]  # gridspacing in t
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number


    def P_j(t):
        return l_boundary_condition(0,t)

    def Q_j(t):
        return r_boundary_condition(L,t)

    if boundary_type == 'dirichlet':
        mat_size = mx-1
        u_j = initial_condition(x[1:mx],0)
        if source is not None:
            def F(t):
                return source(x[1:mx],t)
        else:
            def F(t):
                return 0

        def rhs(t):
            vec = np.zeros(mat_size)
            vec[0] = P_j(t)
            vec[-1] = Q_j(t)
            vec *= lmbda
            return vec + deltat*F(t)

    elif boundary_type == 'neumann':

        mat_size = mx+1

        u_j = initial_condition(x,0)
        if source is not None:
            def F(t):
                return source(x[1:mx],t)
        else:
            def F(t):
                return 0
        def rhs(t):
            vec = np.zeros(mat_size)
            vec[0] = -P_j(t)
            vec[-1] = Q_j(t)
            vec *= 2*lmbda*deltat
            return vec + deltat*F(t)

    elif boundary_type == 'periodic':
        mat_size = mx
        u_j = initial_condition(x[:mx-1],0)
        u_j = np.append(u_j,u_j[-1])

        if source is not None:
            def F(t):
                f = source(x[:mx-1],t)
                f = np.append(f,f[-1])
                return f
        else:
            def F(t):
                return 0

        def rhs(t):
            return deltat * F(t)

    if method == 'forward':
        mat1 = identity(mat_size)
        mat2 = tri_diag(mat_size,lmbda,1-2*lmbda,lmbda)

    elif method == 'backward':
        mat1 = tri_diag(mat_size, -lmbda, 1 + 2 * lmbda, -lmbda)
        mat2 = identity(mat_size)

    elif method == 'crank':
        mat1 = tri_diag(mat_size, -lmbda / 2, 1 + lmbda, -lmbda / 2)
        mat2 = tri_diag(mat_size, lmbda / 2, 1 - lmbda, lmbda / 2)

    if boundary_type == 'neumann':
        mat1[0, 1] *= 2
        mat1[mat_size-1,mat_size-2] *= 2
        mat2[0, 1] *= 2
        mat2[mat_size-1, mat_size - 2] *= 2

    if boundary_type == 'periodic':
        mat1.tolil()
        mat2.tolil()
        mat1[0,mat_size-1] = mat1[0,1]
        mat1[mat_size-1,0] = mat1[0,1]
        mat2[0, mat_size - 1] = mat2[0, 1]
        mat2[mat_size - 1, 0] = mat2[0, 1]
        mat1.tocsr()
        mat2.tocsr()


    for j in range(0, mt):
        vec = rhs(t[j])
        u_j = scipy.sparse.linalg.spsolve(mat1, mat2 * u_j + vec)

    if boundary_type == 'dirichlet':
        u_j = np.concatenate(([l_boundary_condition(0,T)],u_j,[r_boundary_condition(L,T)]))

    if boundary_type == 'periodic':
        u_j = np.append(u_j,u_j[0])

    return x, u_j


def u_exact(x, t,args):
    kappa = args[0]
    L = args[1]
    # the exact solution
    y = \
        np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L) + nu(x,t)
    return y


def l_boundary(x,t):
    return 4

def r_boundary(x,t):
    return 4

def nu(x,t):
    n = x*(r_boundary(x,t)-l_boundary(x,t))/L + l_boundary(x,t)
    return  n



kappa = 1
L = 1
T = 0.5
mx = 10
mt = 1000

xx = np.linspace(0, L, 250)
t = np.linspace(0, T, mt + 1)
exact = u_exact(xx,T,[kappa,L])


#plt.plot(xx,exact,label = 'exact')




def initial(x,t):
    # initial temperature distribution
    y = np.sin(pi * x / L)
    return y


def source(x,t):
    return 10*x


f_x,f_u = solve_diffusive_pde('forward',kappa,L,T,mx,mt,'periodic',l_boundary,r_boundary,initial,source)

b_x,b_u = solve_diffusive_pde('backward',kappa,L,T,mx,mt,'periodic',l_boundary,r_boundary,initial,source)

c_x,c_u = solve_diffusive_pde('crank',kappa,L,T,mx,mt,'periodic',l_boundary,r_boundary,initial,source)

plt.plot(f_x,f_u,label = 'forward')
plt.plot(b_x,b_u,label = 'backward')
plt.plot(c_x,c_u,label = 'crank')

plt.legend()
plt.grid()
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.show()

