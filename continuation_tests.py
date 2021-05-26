import numpy as np
from scipy.optimize import fsolve

from numericalContinuation import continuation
from shooting import shootingG, orbit_shooting

"""
tests for the continuation() function
"""


def input_tests():
    """
    Tests for whether continuation() can handle good/bad input parameters

    It is the expected behaviour that tests using natural parameter continuation will fail for values after the
    bifurcation. This is because at folds (saddle-node bifurcations) the search line does not intersect with the
    curve. However, they are included for illustrative purposes."
    """

    """
    good shooting u0 values
    """
    good_shooting_u0 = np.array([1.4, 0, 6.3])

    """
    good shooting function
    """

    def good_shooting_function(t, u, args):
        beta = args[0]
        sigma = args[1]

        u1 = u[0]
        u2 = u[1]

        du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)
        return np.array([du1dt, du2dt])

    """
    good shooting phase condition
    """

    def good_shooting_pc(u0, args):
        p = good_shooting_function(1, u0, args)[0]
        return p

    """
    good non-shooting u0 values
    """
    good_non_shooting_u0 = np.array([1, 1, 1])

    """
    good non-shooting function
    """

    def good_non_shooting_function(x, args):
        c = args[0]
        return x ** 3 - x + c

    all_tests_passed = True
    failed_tests = []
    """
    Tests Start
    """
    """
    good shooting u0, function and phase condition
    Testing when a good u0 values, function and phase condition are provided for numerical continuation on a shooting
    problem
    """
    """
    Natural parameter
    """
    try:
        continuation('natural', good_shooting_function, good_shooting_u0, [2, -1], 0, [2, 0], 50, shootingG, fsolve,
                     good_shooting_pc)
        print("good shooting u0, func, pc natural: test passed")
    except (TypeError, ValueError):
        all_tests_passed = False
        failed_tests.append("good shooting u0, func, pc natural")
        print("good shooting u0, func, pc natural: test failed")

    """
    Pseudo-arclength
    """
    try:
        continuation('pseudo', good_shooting_function, good_shooting_u0, [2, -1], 0, [2, 0], 50, shootingG, fsolve,
                     good_shooting_pc)
        print("good shooting u0, func, pc pseudo: test passed")
    except (TypeError, ValueError):
        all_tests_passed = False
        failed_tests.append("good shooting u0, func, pc pseduo")
        print("good shooting u0, func, pc pseudo: test failed")

    """
    good non-shooting u0, function and phase condition
    Testing when a good u0 values and function are provided for numerical continuation on a non-shooting
    problem - no errors should be raised
    """
    """
    Natural parameter
    """
    try:
        continuation('natural', good_non_shooting_function, good_non_shooting_u0, [2], 0, [-2, 2], 50, lambda x: x,
                     fsolve, None)
        print("good non-shooting u0, func natural: test passed")
    except (TypeError, ValueError):
        all_tests_passed = False
        failed_tests.append("good non-shooting u0, func natural")
        print("good non-shooting u0, func natural: test failed")

    """
    Pseudo-arclength
    """
    try:
        continuation('pseudo', good_non_shooting_function, good_non_shooting_u0, [2], 0, [-2, 2], 50, lambda x: x,
                     fsolve, None)
        print("good non-shooting u0, func pseudo: test passed")
    except (TypeError, ValueError):
        all_tests_passed = False
        failed_tests.append("good non-shooting u0, func pseudo")
        print("good non-shooting u0, func pseudo: test failed")

    """
    'function' param not a function test
    Tests when 'function' parameter given is not a function - a TypeError should be raised
    """
    try:
        continuation('natural', "not a function", good_shooting_u0, [2, -1], 0, [2, 0], 50, shootingG, fsolve,
                     good_shooting_pc)
        all_tests_passed = False
        failed_tests.append("function not a function")
        print("function not a function: test failed")
    except TypeError:
        print("function not a function: test passed")

    """
    discretisation not a function test
    Tests when discretisation is not a function - a TypeError should be raised
    """
    try:
        continuation('natural', good_shooting_function, good_shooting_u0, [2, -1], 0, [2, 0], 50, "not a function",
                     fsolve, good_shooting_pc)
        all_tests_passed = False
        failed_tests.append("discretisation not a function")
        print("discretisation not a function: test failed")
    except TypeError:
        print("discretisation not a function: test passed")

    """
    discretisation(function) not a function test
    Tests when discretisation(function) does not return a function - a TypeError should be raised
    """

    def bad_discretisation(func):
        return "not a function"

    try:
        continuation('natural', good_shooting_function, good_shooting_u0, [2, -1], 0, [2, 0], 50, bad_discretisation,
                     fsolve, good_shooting_pc)
        all_tests_passed = False
        failed_tests.append("discretisation(function) not a function")
        print("discretisation(function) not a function: test failed")
    except TypeError:
        print("discretisation(function) not a function: test passed")

    """
    function wrong output type test
    Tests when function returns an output of the wrong type - a TypeError should be raised
    """

    def function_wrong_output_type(t, u0, args):
        return "wrong type"

    try:
        continuation('natural', function_wrong_output_type, good_shooting_u0, [2, -1], 0, [2, 0], 50, lambda x: x,
                     fsolve, None)
        all_tests_passed = False
        failed_tests.append("function wrong output type")
        print("function wrong output type: test failed")
    except TypeError:
        print("function wrong output type: test passed")

    """
    function wrong output shape test
    Tests when function returns an output of the wrong shape - a TypeError should be raised
    """

    def function_wrong_output_shape(t, u0, args):
        return [u0, u0]

    try:
        continuation('natural', function_wrong_output_shape, good_shooting_u0, [2, -1], 0, [2, 0], 50, lambda x: x,
                     fsolve, None)
        all_tests_passed = False
        failed_tests.append("function wrong output shape")
        print("function wrong output shape: test failed")
    except TypeError:
        print("function wrong output shape: test passed")

    """
    pc not a function test
    Tests when phase condition given is not None or a function - a TypeError should be raised
    """
    try:
        continuation('natural', good_shooting_function, good_shooting_u0, [2, -1], 0, [2, 0], 50, shootingG, fsolve,
                     "not a function")
        all_tests_passed = False
        failed_tests.append("pc not a function")
        print("pc not a function: test failed")
    except TypeError:
        print("pc not a function: test passed")

    """
    pc wrong output type
    Testing when pc param returns an output of the wrong type - a TypeError should be raised
    """

    def pc_wrong_output_type(u0, args):
        return "wrong type"

    try:
        continuation('natural', good_shooting_function, good_shooting_u0, [2, -1], 0, [2, 0], 50, shootingG, fsolve,
                     pc_wrong_output_type)
        all_tests_passed = False
        failed_tests.append("pc wrong output type")
        print("pc wrong output type: test failed")
    except TypeError:
        print("pc wrong output type: test passed")

    """
    pc wrong output shape 
    Testing when pc param returns an output of the wrong shape - a TypeError should be raise
    """

    def pc_wrong_output_shape(u0, args):
        return [u0, u0]

    try:
        continuation('natural', good_shooting_function, good_shooting_u0, [2, -1], 0, [2, 0], 50, shootingG, fsolve,
                     pc_wrong_output_shape)
        all_tests_passed = False
        failed_tests.append("pc wrong output shape")
        print("pc wrong output shape: test failed")
    except TypeError:
        print("pc wrong output shape: test passed")

    """
    incorrect u0 type
    Testing when the u0 param is the wrong type - a TypeError should be raised
    """
    try:
        continuation('natural', good_shooting_function, "wrong type", [2, -1], 0, [2, 0], 50, shootingG, fsolve,
                     good_shooting_pc)
        all_tests_passed = False
        failed_tests.append("incorrect u0 type")
        print("incorrect u0 type: test failed")
    except TypeError:
        print("incorrect u0 type: test passed")

    """
    incorrect pars type
    Testing when the pars param is the wrong type - a TypeError should be raised
    """
    try:
        continuation('natural', good_shooting_function, good_shooting_u0, "wrong type", 0, [2, 0], 50, shootingG,
                     fsolve, good_shooting_pc)
        all_tests_passed = False
        failed_tests.append("incorrect pars type")
        print("incorrect pars type: test failed")
    except TypeError:
        print("incorrect pars type: test passed")

    """
    incorrect vary_par type
    Testing when the vary_par param is the wrong type - a TypeError should be raised
    """
    try:
        continuation('natural', good_shooting_function, good_shooting_u0, [2, -1], "wrong type", [2, 0], 50, shootingG,
                     fsolve, good_shooting_pc)
        all_tests_passed = False
        failed_tests.append("incorrect vary_par type")
        print("incorrect vary_par type: test failed")
    except TypeError:
        print("incorrect vary_par type: test passed")

    """
    vary_par < 0
    Testing when the vary_par param is an integer but < 0 - a ValueError should be raised
    """
    try:
        continuation('natural', good_shooting_function, good_shooting_u0, [2, -1], -1, [2, 0], 50, shootingG,
                     fsolve, good_shooting_pc)
        all_tests_passed = False
        failed_tests.append("vary_par < 0")
        print("vary_par < 0: test failed")
    except ValueError:
        print("vary_par < 0: test passed")

    """
    incorrect vary_par_number type
    Testing when the vary_par_number param is the wrong type - a TypeError should be raised
    """
    try:
        continuation('natural', good_shooting_function, good_shooting_u0, [2, -1], 0, [2, 0], "wrong type", shootingG,
                     fsolve, good_shooting_pc)
        all_tests_passed = False
        failed_tests.append("incorrect vary_par_number type")
        print("incorrect vary_par_number type: test failed")
    except TypeError:
        print("incorrect vary_par_number type: test passed")

    """
    vary_par_number < 1
    Testing when the vary_par_number param is an integer but < 1 - a ValueError should be raised
    """
    try:
        continuation('natural', good_shooting_function, good_shooting_u0, [2, -1], 0, [2, 0], -1, shootingG,
                     fsolve, good_shooting_pc)
        all_tests_passed = False
        failed_tests.append("vary_par_number < 0")
        print("vary_par_number < 1: test failed")
    except ValueError:
        print("vary_par_number < 1: test passed")

    """
    incorrect vary_par_range type test
    Testing when the vary_par_range param is the wrong type - a TypeError should be raised
    """
    try:
        continuation('natural', good_shooting_function, good_shooting_u0, [2, -1], 0, "wrong type", 50, shootingG,
                     fsolve, good_shooting_pc)
        all_tests_passed = False
        failed_tests.append("incorrect vary_par_range type")
        print("incorrect vary_par_range type: test failed")

    except TypeError:
        print("incorrect vary_par_range type: test passed")

    """
    incorrect vary_par_range length test
    Testing when the vary_par_param is not in the shape [a, b] - a ValueError should be raise
    """
    try:
        continuation('natural', good_shooting_function, good_shooting_u0, [2, -1], 0, [1, 1, 1], 50, shootingG,
                     fsolve, good_shooting_pc)
        all_tests_passed = False
        failed_tests.append("incorrect vary_par_range length")
        print("incorrect vary_par_range length: test failed")

    except ValueError:
        print("incorrect vary_par_range length: test passed")

    """
    incorrect vary_par_range element type
    Testing when an element of vary_par_range is the wrong type - a TypeError should be raised
    """
    try:
        continuation('natural', good_shooting_function, good_shooting_u0, [2, -1], 0, [2, "wrong type"], 50, shootingG,
                     fsolve, good_shooting_pc)
        all_tests_passed = False
        failed_tests.append("incorrect vary_par_range element type")
        print("incorrect vary_par_range element type: test failed")

    except TypeError:
        print("incorrect vary_par_range element type: test passed")

    """
    Results
    """
    if all_tests_passed:
        print("___________")
        print("All input tests passed :)")
        print("___________")
    else:
        print("___________")
        print("Some input tests failed :(")
        print("___________")
        print("Tests Failed:")
        [print(fail + ' test') for fail in failed_tests]


def value_tests():
    """
    Tests for whether continuation() produces correct output values
    """
    all_tests_passed = True
    failed_tests = []
    """
    Tests Start
    """
    """
    hopf normal tests
    """
    """
    Function for the Hopf normal form equations
    """

    def hopf_normal(t, u, args):
        beta = args[0]
        sigma = args[1]

        u1 = u[0]
        u2 = u[1]
        du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)

        du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)
        return np.array([du1dt, du2dt])

    """
    Function for the phase condition for the Hopf normal form equations
    """

    def pc_hopf_normal(u0, args):
        p = hopf_normal(1, u0, args)[0]
        return p

    """
    Function giving the explicit solution of the Hopf normal form equations
    """

    def hopf_normal_explicit(beta, t, phase):
        u1 = np.sqrt(beta) * np.cos(t + phase)
        u2 = np.sqrt(beta) * np.sin(t + phase)
        return np.array([u1, u2])

    """
    Initial values for hopf normal form 
    """
    u0_hopf_normal = np.array([1.4, 0, 6.3])

    """
    hopf normal before bifurcation test
    Testing performance of numerical continuation on the shooting method for the hopf normal form equations by using 
    numerical continuation to estimate the solutions for a list of parameter values then comparing those values to the
    true values. Testing for parameter values of beta  in the interval [2, 0] ie. before the bifurcation. Comparison 
    done to a tolerance of 1e-6.
    """
    """
    Natural parameter continuation
    """
    par_list, x = continuation('natural', hopf_normal, u0_hopf_normal, [2, -1], 0, [2, 0], 100, shootingG, fsolve,
                               pc_hopf_normal)

    true = np.array([hopf_normal_explicit(par, 0, T) for par, T in list(zip(par_list, x[:, -1]))])

    if np.allclose(true, x[:, :-1], atol=1e-6):
        print("hopf normal before bifurcation natural: test passed")
    else:
        all_tests_passed = False
        failed_tests.append("hopf normal before bifurcation natural")
        print("hopf normal before bifurcation natural: test failed")

    """
    Pseudo-arclength continuation
    """
    par_list, x = continuation('pseudo', hopf_normal, u0_hopf_normal, [2, -1], 0, [2, 0], 100, shootingG, fsolve,
                               pc_hopf_normal)

    true = np.array([hopf_normal_explicit(par, 0, T) for par, T in list(zip(par_list, x[:, -1]))])

    if np.allclose(true, x[:, :-1], atol=1e-6):
        print("hopf normal before bifurcation pseudo: test passed")
    else:
        all_tests_passed = False
        failed_tests.append("hopf normal before bifurcation pseudo")
        print("hopf normal before bifurcation pseudo: test failed")

    """
    hopf normal after bifurcation test
    Testing performance of numerical continuation on the shooting method for the hopf normal form equations by using 
    numerical continuation to estimate the solutions for a list of parameter values then comparing those values to the
    true values. Testing for parameter values of beta in the interval [0, -1], ie. after the bifurcation. Comparison 
    done to a tolerance of 1e-6.
    Expected to fail for the natural parameter continuation case.
    """
    """
    Natural parameter continuation
    Expected to fail
    """
    par_list, x = continuation('natural', hopf_normal, u0_hopf_normal, [2, -1], 0, [0, -1], 100, shootingG, fsolve,
                               pc_hopf_normal)

    true = np.array([hopf_normal_explicit(par, 0, T) for par, T in list(zip(par_list, x[:, -1]))])

    if np.allclose(true, x[:, :-1], atol=1e-6):
        print("hopf normal after bifurcation natural: test passed")
    else:
        all_tests_passed = False
        failed_tests.append("hopf normal after bifurcation natural - Expected")
        print("hopf normal after bifurcation natural: test failed - Expected")

    """
    Pseudo-arclength continuation
    """
    par_list, x = continuation('pseudo', hopf_normal, u0_hopf_normal, [2, -1], 0, [0, -1], 100, shootingG, fsolve,
                               pc_hopf_normal)

    true = np.array([hopf_normal_explicit(par, 0, T) for par, T in list(zip(par_list, x[:, -1]))])

    # excludes very last value as that has crossed the y-axis and is not valid.
    if np.allclose(true[:-1], x[:, :-1][:-1], atol=1e-6):
        print("hopf normal after bifurcation pseudo: test passed")
    else:
        all_tests_passed = False
        failed_tests.append("hopf normal after bifurcation pseudo")
        print("hopf normal after bifurcation pseudo: test failed")

    """
    cubic equation tests
    """
    """
    function for the cubic equation
    """

    def cubic(x, args):
        c = args[0]
        return x ** 3 - x + c

    """
    function giving explicit solution for cubic equation
    """

    def cubic_explicit(x, val):
        root = fsolve(cubic, x, [val])
        return root

    """
    Initial values for the cubic equation
    """
    u0_cubic = np.array([1, 1, 1])
    """
    cubic before bifurcation test
    Testing performance of numerical continuation on the cubic equation by using numerical continuation to estimate the 
    solutions for a list of parameter values then comparing those values to the true values. Testing for parameter 
    values of c in the interval [-2, 0.38], ie. before the bifurcation. Comparison done to a tolerance of 1e-6.
    """
    """
    Natural parameter continuation
    """
    par_list, x = continuation('natural', cubic, u0_cubic, [-2], 0, [-2, 0.38], 100, lambda x: x, fsolve, None)

    true = np.array([cubic_explicit(x, par) for par, x in list(zip(par_list, x))])

    if np.allclose(true, x, atol=1e-6):
        print("cubic before bifurcation natural: test passed")
    else:
        all_tests_passed = False
        failed_tests.append("cubic before bifurcation natural")
        print("cubic before bifurcation natural: test failed")

    """
    Pseudo-arclength continuation
    """
    par_list, x = continuation('pseudo', cubic, u0_cubic, [-2], 0, [-2, 0.38], 100, lambda x: x, fsolve, None)

    true = np.array([cubic_explicit(x, par) for par, x in list(zip(par_list, x))])

    if np.allclose(true, x, atol=1e-6):
        print("cubic before bifurcation pseudo: test passed")
    else:
        all_tests_passed = False
        failed_tests.append("cubic before bifurcation pseudo")
        print("cubic before bifurcation pseudo: test failed")

    """
    cubic after bifurcation test
    Testing performance of numerical continuation on the cubic equation by using numerical continuation to estimate the 
    solutions for a list of parameter values then comparing those values to the true values. Testing for parameter 
    values of c in the interval [0.38, 2], ie. after the bifurcation. Comparison done to a tolerance of 1e-6.
    Expected to fail for the natural parameter continuation case.
    """
    """
    Natural parameter continuation
    Expected to fail
    """
    par_list, x = continuation('natural', cubic, u0_cubic, [-2], 0, [0.38, 2], 100, lambda x: x, fsolve, None)

    true = np.array([cubic_explicit(x, par) for par, x in list(zip(par_list, x))])

    if np.allclose(true, x, atol=1e-6):
        print("cubic after bifurcation natural: test passed")
    else:
        all_tests_passed = False
        failed_tests.append("cubic after bifurcation natural - Expected")
        print("cubic after bifurcation natural: test failed - Expected")

    """
    Pseudo-arclength continuation
    """
    par_list, x = continuation('pseudo', cubic, u0_cubic, [-2], 0, [0.38, 2], 100, lambda x: x, fsolve, None)

    true = np.array([cubic_explicit(x, par) for par, x in list(zip(par_list, x))])

    if np.allclose(true, x, atol=1e-6):
        print("cubic after bifurcation pseudo: test passed")
    else:
        all_tests_passed = False
        failed_tests.append("cubic after bifurcation pseudo")
        print("cubic after bifurcation pseudo: test failed")

    """
    Results
    """
    if all_tests_passed:
        print("___________")
        print("All value tests passed :)")
        print("___________")
    else:
        print("___________")
        print("Some value tests failed :(")
        print("___________")
        print("Tests Failed:")
        [print(fail + ' test') for fail in failed_tests]


def main():
    # print("Input Tests:")
    # input_tests()
    print("Value Tests:")
    print("It is the expected behaviour that tests using natural parameter continuation will fail for values after the "
          "bifurcation.\nThis is because at folds (saddle-node bifurcations) the search line does not intersect with "
          "the curve.\nHowever, they are included for illustrative purposes.")
    value_tests()


if __name__ == '__main__':
    main()
