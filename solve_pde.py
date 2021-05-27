import warnings
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from solve_ode import integer_float_array_input_check


def solve_diffusive_pde(method, kappa, L, T, mx, mt, boundary_type, l_boundary_condition_func,
                        r_boundary_condition_func, initial_condition_func,
                        source_func=None, lb_args=None, rb_args=None, ic_args=None, so_args=None):
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
    :param l_boundary_condition_func: Left boundary condition - in form f(x, t, *args)
    :param r_boundary_condition_func: Right boundary condition - in form f(x, t, *args)
    N.B.: for periodic boundary conditions it must be that l_boundary_condition == r_boundary_condition
    :param initial_condition_func: Initial condition to apply - in form f(x, t, *args)
    :param source_func: Source parameter if one is to be used - in form f(x, t, *args)
    :param lb_args: Any args to be passed to l_boundary_condition
    :param rb_args: Any args to be passed to r_boundary_condition
    :param ic_args: Any args to be passed to initial_condition
    :param so_args: Any args to be passed to source
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
        return tri_diag(size, 0, 1, 0)

    def func_in_x_and_t(func, func_args):
        """
        Function to redefine f(x,t, *args) as f(x, t)
        """
        if func_args is None:
            return func
        else:
            def in_x_and_t(x, t):
                return func(x, t, func_args)

            return in_x_and_t

    def param_int_or_float(par_name, param):
        """
        Function checking whether input is an integer or float. Returns a TypeError if not
        """
        if not isinstance(param, (int, float, np.int_, np.float_)):
            raise TypeError(f"{par_name}: {param} needs to be an integer or float")

    def greater_than_zero(par_name, param):
        """
        Function checking whether input is > 0. Returns a ValueError if not
        """
        if not param > 0:
            raise ValueError(f"{par_name}: {param} must be > 0.")

    def function_check(func_name, func, args):
        """
        function for checking l_boundary_condition, r_boundary_condition, initial_condition, and source are functions
        that take the right inputs and return right output type/shape.
        """
        # checks if a function
        if callable(func):

            # checks if it works with correct inputs
            # exception for initial_condition() which should return an array, not a scalar
            if func_name == 'initial_condition_func':
                inp = np.array([1])
                check = integer_float_array_input_check
            else:
                inp = 1
                check = param_int_or_float
            try:
                if args is not None:
                    test = func(inp, inp, args)
                else:
                    test = func(inp, inp)
            except:
                raise TypeError(f"{func_name} must take inputs in the form f(x, t) or have arguments provided.")

            # checks that function returns a float or int or array of float or ints
            check(f"{func_name} output", test)

        else:
            raise TypeError(f"{func_name} is not a function.")

    # checks that kappa is an int or float
    param_int_or_float('kappa', kappa)

    # checks that L is an int or float
    param_int_or_float('L', L)

    # checks that L is > 0
    greater_than_zero('L', L)

    # checks that T is an int or float
    param_int_or_float('T', T)

    # checks that T is > 0
    greater_than_zero('T', T)

    # checks that mx is an integer
    if not isinstance(mx, (int, float, np.float_, np.int_)):
        raise TypeError(f"mx: {mx} is not an integer.")
    elif not float(mx).is_integer():
        raise TypeError(f"mx: {mx} is not an integer.")
    else:
        mx = int(mx)
    # checks that mt is an integer
    if not isinstance(mt, (int, float, np.int_, np.int_)):
        raise TypeError(f"mt: {mt} is not an integer.")
    elif not float(mt).is_integer():
        raise TypeError(f"mt: {mt} is not an integer.")
    else:
        mt = int(mt)

    # checks that mx is > 0
    greater_than_zero('mx', mx)

    # checks that mt is > 0
    greater_than_zero('mt', mt)

    # checks l_boundary_condition is valid
    function_check('l_boundary_condition_func', l_boundary_condition_func, lb_args)

    # checks r_boundary_condition is valid
    function_check('r_boundary_condition_func', r_boundary_condition_func, rb_args)

    # checks initial_condition is valid
    function_check('initial_condition_func', initial_condition_func, ic_args)

    # checks source is valid if not None
    if source_func is not None:
        function_check('source_func', source_func, so_args)

    # redefines l_boundary_condition_func as f(x, t)
    l_boundary_condition = func_in_x_and_t(l_boundary_condition_func, lb_args)

    # redefines r_boundary_condition_func as f(x, t)
    r_boundary_condition = func_in_x_and_t(r_boundary_condition_func, rb_args)

    # redefines initial_condition_func as f(x, t)
    initial_condition = func_in_x_and_t(initial_condition_func, ic_args)

    # redefines source_condition_func as f(x, t)
    source = func_in_x_and_t(source_func, so_args)

    # SciPy doesn't like some sparse matrix operations done but performance isn't affected so can be ignored.
    warnings.filterwarnings('ignore', category=scipy.sparse.SparseEfficiencyWarning)

    """
    Sets up numerical parameters needed
    """
    x = np.linspace(0, L, mx + 1)  # mesh points in space
    t = np.linspace(0, T, mt + 1)  # mesh points in time

    deltax = x[1] - x[0]  # grid spacing in x
    deltat = t[1] - t[0]  # grid spacing in t
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number

    def P_j(t):
        """
        Function giving value of left boundary at time t
        """
        return l_boundary_condition(0, t)

    def Q_j(t):
        """
        Function giving value of right boundary at time t
        """
        return r_boundary_condition(L, t)

    if boundary_type == 'dirichlet':
        mat_size = mx - 1  # dirichlet boundaries need an m-1 x m-1 matrix
        u_j = initial_condition(x[1:mx], 0)  # creates start vector using initial condition function

        # if there is a a source term, create a function F(t) which returns the value of the source at time t
        # else no source --> 0
        if source is not None:
            def F(t):
                return source(x[1:mx], t)
        else:
            def F(t):
                return 0

        # constructs function returning extra RHS vector to be used during solving at time t
        def rhs(t):
            vec = np.zeros(mat_size)
            vec[0] = P_j(t)
            vec[-1] = Q_j(t)
            vec *= lmbda
            return vec + deltat * F(t)  # vec = lmbda*(p_j, 0, 0,..., q_j) + F(x_j, t_j)

    elif boundary_type == 'neumann':
        mat_size = mx + 1  # neumann boundaries need a matrix of size m+1 x m+1
        u_j = initial_condition(x, 0)  # creates start vector using initial condition function

        # if there is a a source term, create a function F(t) which returns the value of the source at time t
        # else no source --> 0
        if source is not None:
            def F(t):
                return source(x, t)
        else:
            def F(t):
                return 0

        # constructs function returning extra RHS vector to be used during solving at time t
        def rhs(t):
            vec = np.zeros(mat_size)
            vec[0] = -P_j(t)
            vec[-1] = Q_j(t)
            vec *= 2 * lmbda * deltat
            return vec + deltat * F(t)  # vec = 2*lmbda*deltat*(-p_j, 0, 0,..., q_j) + F(x_j, t_j)

    elif boundary_type == 'periodic':
        # checks that left and right boundary conditions are the same
        if l_boundary_condition(1, 1) != r_boundary_condition(1, 1):
            raise ValueError("For boundary_type_periodic the left and right boundary conditions must be the same.")
        mat_size = mx  # neumann boundaries need a matrix of size m x m
        u_j = initial_condition(x[:mx - 1], 0)
        u_j = np.append(u_j, u_j[-1])  # u_j = (u_0, u_1,..., u_m-1, u_m-1)

        # if there is a a source term, create a function F(t) which returns the value of the source at time t
        # else no source --> 0
        if source is not None:
            def F(t):
                f = source(x[:mx - 1], t)
                f = np.append(f, f[-1])
                return f
        else:
            def F(t):
                return 0

        # constructs function returning extra RHS vector to be used during solving at time t
        # no vector for periodic, only deltat*F term if source
        def rhs(t):
            return deltat * F(t)

    # raises ValueError if bad boundary_type is given
    else:
        raise ValueError(f"boundary_type: '{boundary_type}' is not valid. Please select 'dirichlet', 'neumann' or"
                         f" 'periodic'.")

    """
    creates sparse matrices dependant on method
    """

    # creates sparse A_FE matrix for forward Euler
    if method == 'forward':
        mat1 = identity(mat_size)
        mat2 = tri_diag(mat_size, lmbda, 1 - 2 * lmbda, lmbda)

    # creates sparse A_BE matrix for backwards EUler
    elif method == 'backward':
        mat1 = tri_diag(mat_size, -lmbda, 1 + 2 * lmbda, -lmbda)
        mat2 = identity(mat_size)

    # creates sparse A_CN and B_CN matrices for Crank-Nicholson
    elif method == 'crank':
        mat1 = tri_diag(mat_size, -lmbda / 2, 1 + lmbda, -lmbda / 2)
        mat2 = tri_diag(mat_size, lmbda / 2, 1 - lmbda, lmbda / 2)

    # raise ValueError is bad method is given
    else:
        raise ValueError(f"method: '{method}' is not valid. Please select 'forward', 'backward', or 'crank'.")

    """
    modifies matrices dependant on boundary condition type
    Dirichlet boundaries need no modification
    """

    # modifies matrices for Neumann boundary conditions
    if boundary_type == 'neumann':
        mat1[0, 1] *= 2
        mat1[mat_size - 1, mat_size - 2] *= 2
        mat2[0, 1] *= 2
        mat2[mat_size - 1, mat_size - 2] *= 2

    # modifies matrices for periodic boundary conditions
    if boundary_type == 'periodic':
        mat1[0, mat_size - 1] = mat1[0, 1]
        mat1[mat_size - 1, 0] = mat1[0, 1]
        mat2[0, mat_size - 1] = mat2[0, 1]
        mat2[mat_size - 1, 0] = mat2[0, 1]

    # Solves the PDE: loop over all time points
    for j in range(0, mt):
        # gets RHS vector for time t_j
        vec = rhs(t[j])

        # solves matrix equation to get updated u_j
        u_j = scipy.sparse.linalg.spsolve(mat1, mat2 * u_j + vec)

    # adds boundary conditions to start and end if Dirichlet boundary condition
    if boundary_type == 'dirichlet':
        u_j = np.concatenate(([l_boundary_condition(0, T)], u_j, [r_boundary_condition(L, T)]))

    # sets u_m = u_0 if periodic boundary condition
    if boundary_type == 'periodic':
        u_j = np.append(u_j, u_j[0])

    # returns x values and solution at T
    return x, u_j


def main():
    """
    Example 1: simple 1D heat equation
    """

    """
    values for heat equation
    """
    kappa = 1
    L = 2
    T = 0.5
    mx = 100
    mt = 1000

    """
    boundary conditions u(0,t) = 0, u(L,t) = 0
    """

    def l_boundary(x, t):
        return 0

    def r_boundary(x, t):
        return 0

    """
    initial condition
    """

    def initial(x, t, L):
        # initial temperature distribution
        y = np.sin(pi * x / L)
        return y

    """
    exact solution
    """

    def heat_1D_exact(x, t, kappa, L):
        # the exact solution
        y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
        return y

    """
    plotting exact solution
    """
    xx = np.linspace(0, L, 250)
    exact = heat_1D_exact(xx, T, kappa, L)
    plt.plot(xx, exact, label='exact')

    """
    approximating solution using 3 methods - forward Euler, backward Euler, Crank-Nicholson
    """
    f_x, f_u = solve_diffusive_pde('forward', kappa, L, T, mx, mt, 'dirichlet', l_boundary, r_boundary, initial,
                                   ic_args=L)

    b_x, b_u = solve_diffusive_pde('backward', kappa, L, T, mx, mt, 'dirichlet', l_boundary, r_boundary, initial,
                                   ic_args=L)

    c_x, c_u = solve_diffusive_pde('crank', kappa, L, T, mx, mt, 'dirichlet', l_boundary, r_boundary, initial,
                                   ic_args=L)

    """
    plotting approximate solutions
    """
    plt.plot(f_x, f_u, label='forward')
    plt.plot(b_x, b_u, label='backward')
    plt.plot(c_x, c_u, label='crank')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x,0.5)')
    plt.show()

    """
    It can be seen that all 3 methods provide a good approximation of the true solution.
    """

    """
    Example 2: Neumann boundary conditions with source term
    """
    kappa = 1
    L = 2
    T = 0.5
    mx = 100
    mt = 1000

    """
    boundary conditions du/dx(0, t) = t, du/dx(L, t) = 1
    """

    def l_boundary(x, t):
        return 0

    def r_boundary(x, t):
        return 1

    """
    initial condition
    """

    def initial(x, t, L):
        # initial temperature distribution
        y = np.sin(pi * x / L)
        return y

    """
    source term F(x,t) = x + t
    """

    def source(x, t):
        return x + t

    """
    approximating solution using 3 methods - forward Euler, backward Euler, Crank-Nicholson
    """
    f_x, f_u = solve_diffusive_pde('forward', kappa, L, T, mx, mt, 'neumann', l_boundary, r_boundary, initial,
                                   source, ic_args=L)

    b_x, b_u = solve_diffusive_pde('backward', kappa, L, T, mx, mt, 'neumann', l_boundary, r_boundary, initial,
                                   source, ic_args=L)

    c_x, c_u = solve_diffusive_pde('crank', kappa, L, T, mx, mt, 'neumann', l_boundary, r_boundary, initial,
                                   source, ic_args=L)

    """
    plotting approximate solutions
    """
    plt.plot(f_x, f_u, label='forward')
    plt.plot(b_x, b_u, label='backward')
    plt.plot(c_x, c_u, label='crank')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x,0.5)')
    plt.show()

    """
    There is not a known exact solution to compare against this time. However, the shape of the approximate solution
    curve makes sense given the physical interpretation of the conditions, so I am very sure it is correct.
    """

    """
    Numerical continuation can be performed on PDEs to explore how their dynamics change as a parameter is varied.
    2 examples of this are shown at the end of numericalContinuation.main()
    """



if __name__ == '__main__':
    main()
