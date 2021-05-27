from math import pi

import numpy as np

from solve_pde import solve_diffusive_pde

"""
tests for the solve_diffusive_pde() function
"""


def input_tests():
    """
    Tests for whether solve_diffusive_pde() can handle good/bad input parameters:
    """

    """
    good functions
    """

    def good_l_boundary_no_args(x, t):
        return 0

    def good_r_boundary_no_args(x, t):
        return 0

    def good_initial_condition_no_args(x, t):
        return 3 * x

    def good_source_no_args(x, t):
        return 2

    def good_l_boundary_args(x, t, args):
        return args

    def good_r_boundary_args(x, t, args):
        return args

    def good_initial_condition_args(x, t, L):
        return np.sin(pi * x / L)

    def good_source_args(x, t, args):
        return args

    """
    bad functions
    """

    def bad_return_type(x, t):
        return 'wrong type'

    def bad_input_shape(x, t, p, q):
        return 0

    all_tests_passed = True
    failed_tests = []
    """
    Tests Start
    """
    """
    all good no args no source test
    Testing when all parameters provided are good and none require args. No source provided - no errors should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args,
                            good_initial_condition_no_args)
        print("all good no args no source: test passed")
    except (TypeError, ValueError):
        all_tests_passed = False
        failed_tests.append("all good no args no source")
        print("all good no args no source: test failed")

    """
    all good no args source test
    Testing when all parameters provided are good and none require args. Source provided. No errors should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args,
                            good_initial_condition_no_args, good_source_no_args)
        print("all good no args source: test passed")
    except (TypeError, ValueError):
        all_tests_passed = False
        failed_tests.append("all good no args source")
        print("all good no args source: test failed")

    """
    all good lb args no source test
    Testing when all parameters provided are good and left boundary requires args. No Source. No errors should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_args,
                            good_r_boundary_no_args,
                            good_initial_condition_no_args, lb_args=0)
        print("all good lb args no source: test passed")
    except (TypeError, ValueError):
        all_tests_passed = False
        failed_tests.append("all good lb args no source")
        print("all good lb args no source: test failed")

    """
    all good rb args no source test
    Testing when all parameters provided are good and right boundary requires args. No Source. No errors should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_args,
                            good_initial_condition_no_args, rb_args=0)
        print("all good rb args no source: test passed")
    except (TypeError, ValueError):
        all_tests_passed = False
        failed_tests.append("all good rb args no source")
        print("all good rb args no source: test failed")

    """
    all good ic args no source test
    Testing when all parameters provided are good and initial condition requires args. No Source. 
    No errors should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args,
                            good_initial_condition_args, ic_args=1)
        print("all good ic args no source: test passed")
    except (TypeError, ValueError):
        all_tests_passed = False
        failed_tests.append("all good ic args no source")
        print("all good ic args no source: test failed")

    """
    all good lb args source test
    Testing when all parameters provided are good and left boundary requires args. Source. No errors should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_args,
                            good_r_boundary_no_args,
                            good_initial_condition_no_args, good_source_no_args, lb_args=0)
        print("all good lb args source: test passed")
    except (TypeError, ValueError):
        all_tests_passed = False
        failed_tests.append("all good lb args source")
        print("all good lb args source: test failed")

    """
    all good rb args source test
    Testing when all parameters provided are good and right boundary requires args. Source. No errors should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_args,
                            good_initial_condition_no_args, good_source_no_args, rb_args=0)
        print("all good rb args source: test passed")
    except (TypeError, ValueError):
        all_tests_passed = False
        failed_tests.append("all good rb args source")
        print("all good rb args source: test failed")

    """
    all good ic args source test
    Testing when all parameters provided are good and initial condition requires args. Source. 
    No errors should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args,
                            good_initial_condition_args, good_source_no_args, ic_args=1)
        print("all good ic args source: test passed")
    except (TypeError, ValueError):
        all_tests_passed = False
        failed_tests.append("all good ic args source")
        print("all good ic args source: test failed")

    """
    all good so args source test
    Testing when all parameters provided are good and source requires args. Source. 
    No errors should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args,
                            good_initial_condition_no_args, good_source_args, so_args=1)
        print("all good so args source: test passed")
    except (TypeError, ValueError):
        all_tests_passed = False
        failed_tests.append("all good so args source")
        print("all good so args source: test failed")

    """
    wrong kappa type test
    Testing when kappa is not a float or int - a TypeError should be raised
    """
    try:
        solve_diffusive_pde('backward', "wrong type", 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("wrong kappa type")
        print("wrong kappa type: test failed")
    except TypeError:
        print("wrong kappa type: test passed")

    """
    wrong L type test
    Testing when L is not a float or int - a TypeError should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, "wrong type", 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("wrong L type")
        print("wrong L type: test failed")
    except TypeError:
        print("wrong L type: test passed")

    """
    wrong T type test
    Testing when T is not a float or int - a TypeError should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, 1, "wrong type", 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("wrong T type")
        print("wrong T type: test failed")
    except TypeError:
        print("wrong T type: test passed")

    """
    wrong mx type test
    Testing when mx is not an  int - a TypeError should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, 1, 1, 1.5, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("wrong mx type")
        print("wrong mx type: test failed")
    except TypeError:
        print("wrong mx type: test passed")

    """
    wrong mt type test
    Testing when mt is not an  int - a TypeError should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, 1, 1, 10, [1000], 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("wrong mt type")
        print("wrong mt type: test failed")
    except TypeError:
        print("wrong mt type: test passed")

    """
    L < 0 test
    Testing when L is < 0 - a ValueError should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, -1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("L < 0")
        print("L < 0: test failed")
    except ValueError:
        print("L < 0: test passed")

    """
    T < 0 test
    Testing when T is < 0 - a ValueError should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, 1, -1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("T < 0")
        print("T < 0: test failed")
    except ValueError:
        print("T < 0: test passed")

    """
    mx < 0 test
    Testing when mx is < 0 - a ValueError should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, 1, 1, -10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("mx < 0")
        print("mx < 0: test failed")
    except ValueError:
        print("mx < 0: test passed")

    """
    mt < 0 test
    Testing when mt is < 0 - a ValueError should be raised
    """
    try:
        solve_diffusive_pde('backward', 1, 1, 1, 10, -1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("mt < 0")
        print("mt < 0: test failed")
    except ValueError:
        print("mt < 0: test passed")

    """
    invalid method test
    Testing when an invalid method is provided - a ValueError should be raised
    """
    try:
        solve_diffusive_pde('not real method', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("invalid method")
        print("invalid method: test failed")
    except ValueError:
        print("invalid method: test passed")

    """
    invalid boundary type test
    Testing when an invalid boundary_type is provided - a ValueError should be raised
    """
    try:
        solve_diffusive_pde('crank', 1, 1, 1, 10, 1000, 'not real type', good_l_boundary_no_args,
                            good_r_boundary_no_args, good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("invalid boundary type")
        print("invalid boundary type: test failed")
    except ValueError:
        print("invalid boundary type: test passed")

    """
    lb not f test
    Testing when l_boundary_func is not a function -  a TypeError should be raised
    """
    try:
        solve_diffusive_pde('crank', 1, 1, 1, 10, 1000, 'dirichlet', "not a function",
                            good_r_boundary_no_args, good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("lb not f")
        print("lb not f: test failed")
    except TypeError:
        print("lb not f: test passed")

    """
    rb not f test
    Testing when r_boundary_func is not a function -  a TypeError should be raised
    """
    try:
        solve_diffusive_pde('crank', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            "not a function", good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("rb not f")
        print("r boundary not f: test failed")
    except TypeError:
        print("r boundary not f: test passed")

    """
    ic not f test
    Testing when initial_condition_func is not a function -  a TypeError should be raised
    """
    try:
        solve_diffusive_pde('crank', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, "not a func")
        all_tests_passed = False
        failed_tests.append("ic not f")
        print("ic not f: test failed")
    except TypeError:
        print("ic not f: test passed")

    """
    so not f test
    Testing when source_func is not a function -  a TypeError should be raised
    """
    try:
        solve_diffusive_pde('crank', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, good_initial_condition_no_args, "not a func")
        all_tests_passed = False
        failed_tests.append("so not f")
        print("so not f: test failed")
    except TypeError:
        print("so not f: test passed")

    """
    lb wrong output type test
    Testing when l_boundary_func returns an output of the wrong type - a TypeError should be raised
    """
    try:
        solve_diffusive_pde('crank', 1, 1, 1, 10, 1000, 'dirichlet', bad_return_type,
                            good_r_boundary_no_args, good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("lb wrong output type")
        print("lb wrong output type: test failed")
    except TypeError:
        print("lb wrong output type: test passed")

    """
    rb wrong output type test
    Testing when r_boundary_func returns an output of the wrong type - a TypeError should be raised
    """
    try:
        solve_diffusive_pde('crank', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            bad_return_type(), good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("rb wrong output type")
        print("rb wrong output type: test failed")
    except TypeError:
        print("rb wrong output type: test passed")

    """
    ic wrong output type test
    Testing when initial_condition_func returns an output of the wrong type - a TypeError should be raised
    """
    try:
        solve_diffusive_pde('crank', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, bad_return_type)
        all_tests_passed = False
        failed_tests.append("ic wrong output type")
        print("ic wrong output type: test failed")
    except TypeError:
        print("ic wrong output type: test passed")

    """
    so wrong output type test
    Testing when source_func returns an output of the wrong type - a TypeError should be raised
    """
    try:
        solve_diffusive_pde('crank', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, good_initial_condition_no_args, bad_return_type)
        all_tests_passed = False
        failed_tests.append("so wrong output type")
        print("so wrong output type: test failed")
    except TypeError:
        print("so wrong output type: test passed")

    """
    lb wrong input shape test
    Testing when l_boundary_func has an input of the wrong shape - a TypeError should be raised
    """
    try:
        solve_diffusive_pde('crank', 1, 1, 1, 10, 1000, 'dirichlet', bad_input_shape,
                            good_r_boundary_no_args, good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("lb wrong input shape")
        print("lb wrong input shape: test failed")
    except TypeError:
        print("lb wrong input shape: test passed")

    """
    rb wrong input shape test
    Testing when r_boundary_func has an input of the wrong shape - a TypeError should be raised
    """
    try:
        solve_diffusive_pde('crank', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            bad_input_shape(), good_initial_condition_no_args)
        all_tests_passed = False
        failed_tests.append("rb wrong input shape")
        print("rb wrong input shape: test failed")
    except TypeError:
        print("rb wrong input shape: test passed")

    """
    ic wrong input shape test
    Testing when initial_condition_func has an input of the wrong shape - a TypeError should be raised
    """
    try:
        solve_diffusive_pde('crank', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, bad_input_shape)
        all_tests_passed = False
        failed_tests.append("ic wrong input shape")
        print("ic wrong input shape: test failed")
    except TypeError:
        print("ic wrong input shape: test passed")

    """
    so wrong input shape test
    Testing when source_func has an input of the wrong shape - a TypeError should be raised
    """
    try:
        solve_diffusive_pde('crank', 1, 1, 1, 10, 1000, 'dirichlet', good_l_boundary_no_args,
                            good_r_boundary_no_args, good_initial_condition_no_args, bad_input_shape)
        all_tests_passed = False
        failed_tests.append("so wrong input shape")
        print("so wrong input shape: test failed")
    except TypeError:
        print("so wrong input shape: test passed")

    """
    p boundary lb != rb test
    Testing when the left and right boundaries are not the same for a periodic boundary type.
    A ValueError should be raised.
    """
    try:
        solve_diffusive_pde('crank', 1, 1, 1, 10, 1000, 'periodic', good_l_boundary_args,
                            good_r_boundary_args, good_initial_condition_no_args, lb_args=1, rb_args=0)
        all_tests_passed = False
        failed_tests.append("p boundary lb != rb")
        print("p boundary lb != rb: test failed")
    except ValueError:
        print("p boundary lb != rb: test passed")

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
    Tests for whether solve_diffusive_pde() produces correct output values
    """







    all_tests_passed = True
    failed_tests = []
    """
    Tests Start
    """
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
    print("Input Tests:")
    input_tests()
    print("Value Tests:")
    value_tests()


if __name__ == '__main__':
    main()
