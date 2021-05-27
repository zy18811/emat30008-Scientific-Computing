import numpy as np
import pylab as pl
from math import pi
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import warnings


def solve_diffusive_pde(method,kappa, L, T, mx, mt, boundary_type,l_boundary_condition,r_boundary_condition,
                        initial_condition,source = None):
    """
    Solves a diffusive, parabolic PDE with given conditions.
    :param method: Solving scheme to use - 'forward' for forward Euler, 'backward' for backwards Euler, or 'crank' for
    Crank-Nicholson
    :param kappa: PDE Kappa value
    :param L: Length to solve over
    :param T: Time to solve over
    :param mx: Number of grid points in space
    :param mt: Number of grid points in time
    :param boundary_type: Boundary condition type - 'dirichlet' for Dirichlet boundary conditions, 'neumann' for
    Neumann boundary conditions, or 'periodic' for periodic boundary conditions.
    :param l_boundary_condition: Left boundary condition - in form f(x, t)
    :param r_boundary_condition: Right boundary condition - in form f(x, t)
    N.B.: for periodic boundary conditions it must be that l_boundary_condition == r_boundary_condition
    :param initial_condition: Initial condition to apply - in form f(x, t)
    :param source: Source parameter if one is to be used - in form f(x, t)
    :return: Returns solution for PDE over length L at time T.
    """

    def tri_diag(size, diag1, diag2, diag3):
        """
        Returns a tridiagonal, sparse matrix of given size and diagonal values. Values are uniform down each diagonal.
        :param size: Matrix size - produces a square size x size matrix
        :param diag1: Values for the first diagonal
        :param diag2: Values for the second diagonal
        :param diag3: Values for the third diagonal
        :return: Returns a 'csc' format sparse matrix.
        """
        diagonals = np.array([diag1 * np.ones(size - 1), diag2 * np.ones(size), diag3 * np.ones(size - 1)],
                             dtype=object)
        offset = [-1, 0, 1]
        return scipy.sparse.diags(diagonals, offset, format='csc')

    def identity(size):
        """
        Function returning an identity matrix of size 'size'
        """
        return tri_diag(size,0,1,0)

    # SciPy doesn't like some sparse matrix operations done but performance isn't affected so can be ignored.
    warnings.filterwarnings('ignore',category=scipy.sparse.SparseEfficiencyWarning)

    """
    Sets up numerical parameters needed
    """
    x = np.linspace(0, L, mx + 1)  # mesh points in space
    t = np.linspace(0, T, mt + 1)  # mesh points in time

    deltax = x[1] - x[0]  # gridspacing in x
    deltat = t[1] - t[0]  # gridspacing in t
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number


    def P_j(t):
        """
        Function giving value of left boundary at time t
        """
        return l_boundary_condition(0,t)

    def Q_j(t):
        """
        Function giving value of right boundary at time t
        """
        return r_boundary_condition(L,t)

    if boundary_type == 'dirichlet':
        mat_size = mx-1    # dirichlet boundaries need an m-1 x m-1 matrix
        u_j = initial_condition(x[1:mx],0)    # creates start vector using initial condition function

        # if there is a a source term, create a function F(t) which returns the value of the source at time t
        # else no source --> 0
        if source is not None:
            def F(t):
                return source(x[1:mx],t)
        else:
            def F(t):
                return 0

        # constructs function returning extra RHS vector to be used during solving at time t
        def rhs(t):
            vec = np.zeros(mat_size)
            vec[0] = P_j(t)
            vec[-1] = Q_j(t)
            vec *= lmbda
            return vec + deltat*F(t)    # vec = lmbda*(p_j, 0, 0,..., q_j) + F(x_j, t_j)

    elif boundary_type == 'neumann':
        mat_size = mx+1    # neumann boundaries need a matrix of size m+1 x m+1
        u_j = initial_condition(x,0)    # creates start vector using initial condition function

        # if there is a a source term, create a function F(t) which returns the value of the source at time t
        # else no source --> 0
        if source is not None:
            def F(t):
                return source(x[1:mx],t)
        else:
            def F(t):
                return 0

        # constructs function returning extra RHS vector to be used during solving at time t
        def rhs(t):
            vec = np.zeros(mat_size)
            vec[0] = -P_j(t)
            vec[-1] = Q_j(t)
            vec *= 2*lmbda*deltat
            return vec + deltat*F(t)    # vec = 2*lmbda*deltat*(-p_j, 0, 0,..., q_j) + F(x_j, t_j)

    elif boundary_type == 'periodic':
        mat_size = mx   # neumann boundaries need a matrix of size m x m
        u_j = initial_condition(x[:mx-1],0)
        u_j = np.append(u_j,u_j[-1])    # u_j = (u_0, u_1,..., u_m-1, u_m-1)

        # if there is a a source term, create a function F(t) which returns the value of the source at time t
        # else no source --> 0
        if source is not None:
            def F(t):
                f = source(x[:mx-1],t)
                f = np.append(f,f[-1])
                return f
        else:
            def F(t):
                return 0

        # constructs function returning extra RHS vector to be used during solving at time t
        # no vector for periodic, only deltat*F term if source
        def rhs(t):
            return deltat * F(t)

    """
    creates sparse matrices dependant on method
    """

    # creates sparse A_FE matrix for forward Euler
    if method == 'forward':
        mat1 = identity(mat_size)
        mat2 = tri_diag(mat_size,lmbda,1-2*lmbda,lmbda)

    # creates sparse A_BE matrix for backwards EUler
    elif method == 'backward':
        mat1 = tri_diag(mat_size, -lmbda, 1 + 2 * lmbda, -lmbda)
        mat2 = identity(mat_size)

    # creates sparse A_CN and B_CN matrices for Crank-Nicholson
    elif method == 'crank':
        mat1 = tri_diag(mat_size, -lmbda / 2, 1 + lmbda, -lmbda / 2)
        mat2 = tri_diag(mat_size, lmbda / 2, 1 - lmbda, lmbda / 2)

    """
    modifies matrices dependant on boundary condition type
    Dirichlet boundaries need no modification
    """

    # modifies matrices for Neumann boundary conditions
    if boundary_type == 'neumann':
        mat1[0, 1] *= 2
        mat1[mat_size-1,mat_size-2] *= 2
        mat2[0, 1] *= 2
        mat2[mat_size-1, mat_size - 2] *= 2

    # modifies matrices for periodic boundary conditions
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

