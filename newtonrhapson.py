import numpy as np
from scipy.linalg import solve

"""
Implementation of multivariate Newton-Raphson method for finding roots of vector functions.
Does work, but very unstable. It is suggested SciPys fsolve() is used instead of this for root finding elsewhere in 
the code as it performs better. 
"""


def approxJ(f, x, tol=1e-8, *args):
    """
    Numerical approximation of the Jacobian of the given function at the point x.
    :param f: Function to have its Jacobian approximated.
    :param x: Values to approximate Jacobian at
    :param tol: Tolerance
    :param args: Any additional args to be passed to function
    :return: Returns numerical approximation of the Jacobian for a given function at x.
    """
    n = len(x)
    func = f(x, *args)
    jac = np.zeros((n, n))
    for j in range(n):
        Dxj = (abs(x[j]) * tol) if x[j] != 0 else tol
        x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(x)]
        jac[:, j] = (f(x_plus, *args) - func) / Dxj
    return jac


def newtonIter(f, x0, *args):
    """
    Performs one iteration of the Newton-Raphson method on a given function with initial approximation x0.
    :param f: Function to apply method to.
    :param x0: Starting point for iteration
    :param args: Any additional args to be passed to function
    :return: Returns updated approximation x1 yielded by applying one iteration of the method to x0.
    """
    J = approxJ(f, x0, 1e-8, *args)  # Approximates Jacobian

    """
    Solves J(x1-x0) = -F(x0) for x1. Raises error if Jacobian is singular.
    """
    try:
        x1_minus_x0 = solve(J, -f(x0, *args))
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError(f"Singular Jacobian --> Initial guess has caused solution to diverge.\n"
                                    "Please try again with a different initial guess.")
    x1 = x1_minus_x0 + x0
    return x1


def newton(f, x0, args):
    """
    Multivariate Newton-Raphson method to find the root of a vector function. Takes function and initial approximation
    of root, and returns root.
    :param f: Function to find the root of
    :param x0: Initial approximation of root
    :param args: Any additional args to be passed to the function
    :return:
    """

    """
    Performs Newton-Raphson iterations, starting from initial approximation. Breaks if value of f() is close to zero
    indicating a root has been found.
    """
    iter_count = 0
    while True:
        check = f(x0, *args)  # Value of function at x
        zero = np.zeros(np.shape(check))  # Generates array of zeros
        if np.allclose(check, zero):  # Checks if value of function is close to zero --> root. Breaks if root is found
            break
        x0 = newtonIter(f, x0, *args)  # Applies an iteration to function at current root approximation
        iter_count += 1

        # Halts if no root is found after 1000 iterations and returns an empty array
        if iter_count == 1000:
            return []
    return x0
