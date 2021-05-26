import numpy as np
from solve_ode import solve_ode

"""
tests for the solve_ode function
"""


def input_tests():
    """
    array of t values in correct format
    """
    good_t = np.linspace(0, 20, 50)

    """
    correct example of single ODE: dx/dt = a*x
    """

    def single_good_f(t, x, args):
        dxdt = args * x
        return dxdt

    """
    correct example of system of ODEs: dx/dt = a*y, dy/dt = b*x
    """

    def system_good_f(t, u, args):
        x = u[0]
        y = u[1]
        dxdt = args[0] * y
        dydt = args[1] * x
        return np.array([dxdt, dydt])

    """
    function with output of wrong type - should be int, float, list or ndarray
    """

    def f_wrong_output_type(t, x, args):
        return "wrong type"

    """
    function with output of wrong shape - output should be 1D for single ODE
    """

    def f_wrong_output_shape_single(t, x, args):
        return [1, 1]

    """
    function with output of wrong shape - output should be 2D for system of 2 ODEs
    """

    def f_wrong_output_shape_system(t, x, args):
        return 1

    all_tests_passed = True

    """
    correct f - single test
    Testing when good single ODE input - no error should be raised
    """
    try:
        solve_ode(single_good_f, [1], good_t, 'euler', [0.01], False, 1)
        print("correct single f: test passed")
    except TypeError:
        all_tests_passed = False
        print("correct single f: test failed")

    """
    correct f - system test
    Testing when good system of ODEs input - no error should be raised
    """
    try:
        solve_ode(system_good_f, [1, 0], good_t, 'euler', [0.01], True, [1, -1])
        print("correct system f: test passed")
    except TypeError:
        all_tests_passed = False
        print("correct system f: test failed")

    """
    f not function test
    Testing when f param is not a function - a TypeError should be raised
    """
    try:
        solve_ode("not a function", [1], good_t, 'euler', [0.01], False, 1)
        all_tests_passed = False
        print("f not a function test: failed")
    except TypeError:
        print("f not a function test: passed")

    """
    f wrong output type test
    Testing when f param returns an output of the wrong type - a TypeError should be raised
    """
    try:
        solve_ode(f_wrong_output_type, [1], good_t, 'euler', [0.01], False, 1)
        all_tests_passed = False
        print("f wrong output type: failed")
    except TypeError:
        print("f wrong output type: passed")

    """
    single f wrong output shape test
    Testing when a single ODE f has an output of the wrong shape - a ValueError should be raised
    """
    try:
        solve_ode(f_wrong_output_shape_single, [1], good_t, 'euler', [0.01], False, 1)
        all_tests_passed = False
        print("single f wrong output shape: failed")
    except ValueError:
        print("single f wrong output shape: passed")

    """
    system f wrong output shape test
    Testing when a system of ODEs f has an output of the wrong shape - a ValueError should be raised
    """
    try:
        solve_ode(f_wrong_output_shape_system, [1, 1], good_t, 'euler', [0.01], True, 1)
        all_tests_passed = False
        print("system f wrong output shape: failed")
    except ValueError:
        print("system f wrong output shape: passed")

    """
    correct x0 type test
    Testing when x0 is the correct type - no error should be raised
    """
    try:
        solve_ode(single_good_f, 1, good_t, 'euler', [0.01], False, 1)
        print("correct x0 type: test passed")
    except TypeError:
        all_tests_passed = False
        print("correct x0 type: test failed")

    """
    incorrect x0 type test
    Testing when x0 is the incorrect type - a TypeError should be raised
    """
    try:
        solve_ode(single_good_f, "wrong x0 type", good_t, 'euler', [0.01], False, 1)
        all_tests_passed = False
        print("incorrect x0 type: test failed")
    except TypeError:
        print("incorrect x0 type: test passed")

    """
    correct t_arr type test
    Testing when t_arr is the correct type - no error should be raised
    """
    try:
        solve_ode(single_good_f, 1, good_t, 'euler', [0.01], False, 1)
        print("correct t_arr type: test passed")
    except TypeError:
        all_tests_passed = False
        print("correct t_arr type: test failed")

    """
    incorrect t_arr type test
    Testing when t_arr is the incorrect type - a TypeError should be raised
    """
    try:
        solve_ode(single_good_f, 1, "wrong t_arr type", 'euler', [0.01], False, 1)
        all_tests_passed = False
        print("incorrect t_arr type: test failed")
    except TypeError:
        print("incorrect t_arr type: test passed")

    """
    correct deltat_max type test
    Testing when deltat_max is the correct type - no error should be raised
    """
    try:
        solve_ode(single_good_f, 1, good_t, 'euler', 0.01, False, 1)
        print("correct deltat_max type: test passed")
    except TypeError:
        all_tests_passed = False
        print("correct deltat_max type: test failed")

    """
    incorrect deltat_max type test
    Testing when deltat_max is the incorrect type - a TypeError should be raised
    """
    try:
        solve_ode(single_good_f, 1, good_t, 'euler', "wrong deltat_max type", False, 1)
        all_tests_passed = False
        print("incorrect deltat_max type: test failed")
    except TypeError:
        print("incorrect deltat_max type: test passed")

    """
    correct method euler test
    Testing when the euler method is chosen correctly - no error should be raised
    """
    try:
        solve_ode(single_good_f, 1, good_t, 'euler', 0.01, False, 1)
        print("correct method euler: test passed")
    except ValueError:
        all_tests_passed = False
        print("correct method euler: test failed")

    """
    correct method rk4 test
    Testing when the rk4 method is chosen correctly - no error should be raised
    """
    try:
        solve_ode(single_good_f, 1, good_t, 'rk4', 0.01, False, 1)
        print("correct method rk4: test passed")
    except ValueError:
        all_tests_passed = False
        print("correct method rk4: test failed")

    """
    incorrect method test
    Testing when an incorrect method is chosen - a ValueError should be raised
    """
    try:
        solve_ode(single_good_f, 1, good_t, 'incorrect method', 0.01, False, 1)
        all_tests_passed = False
        print("incorrect method: test failed")
    except ValueError:
        print("incorrect method: test passed")

    """
    correct system type test
    Testing with a system param of the correct type, boolean - no error should be raised
    """
    try:
        solve_ode(single_good_f, 1, good_t, 'euler', 0.01, False, 1)
        print("correct system type: test passed")
    except TypeError:
        all_tests_passed = False
        print("correct system type: test failed")

    """
    incorrect system type test
    Testing with a system param of an incorrect type - a TypeError should be raised
    """
    try:
        solve_ode(single_good_f, 1, good_t, 'euler', good_t, "wrong system type", 1)
        all_tests_passed = False
        print("incorrect system type: test failed")
    except TypeError:
        print("incorrect system type: test passed")

    if all_tests_passed:
        print("___________")
        print("All input tests passed :)")
        print("___________")
    else:
        print("___________")
        print("Some input tests failed :(")
        print("___________")


def value_tests():
    """
    array of t values in correct format
    """
    good_t = np.linspace(0, 1, 50)

    """
    correct example of single ODE
    """

    def single_good_f(t, x, args):
        dxdt = args * x
        return dxdt

    """
    correct example of system of ODEs
    """

    def system_good_f(t, u, args):
        x = u[0]
        y = u[1]
        dxdt = args[0] * y
        dydt = args[1] * x
        return np.array([dxdt, dydt])

    """
    true solution for single_good_f() to test against
    """

    def single_good_f_true(t, args):
        return np.exp(args * t)

    """
    true solution for system_good_f() to test against
    """

    def system_good_f_true(t, args):
        args = np.abs(args)
        a = np.sqrt(args[0] / args[1])
        b = np.sqrt(args[0] * args[1])
        x = a * np.sin(b * t) + np.cos(b * t)
        y = np.cos(b * t) - a * np.sin(b * t)
        return np.array([x, y])

    """
    Function to test whether a method reaches a given tolerance with a specified step size
    """

    def accuracy_test_single(method, system, x0, step_size_multiplier, tol, function, true_function, initial_step_size,
                             args):
        """
        :param method: Method to use - 'euler' or 'rk4'
        :param system: Boolean for if system of ODEs
        :param x0: Initial conditions
        :param step_size_multiplier: Number to multiply step size by each iteration, 0.1 suggested
        :param tol: Tolerance to test against
        :param function: Function to estimate solution of
        :param true_function: Function returning the true solution of 'function'
        :param initial_step_size: Initial step size value to use
        :param args: Any args to be passed to function
        """
        step_size = initial_step_size
        while True:
            true_sol = true_function(1, args)
            if not system:
                estimated_sol = solve_ode(function, x0, good_t, method, step_size, system, args)[-1]
                if np.isclose(estimated_sol, true_sol, tol):
                    print(f"{method} accurate to a tolerance of {tol} for a single ODE with a step size of {step_size}")
                    break
            else:
                estimated_sol = solve_ode(function, x0, good_t, method, step_size, system, args)[:, -1]
                if np.allclose(estimated_sol, true_sol, tol):
                    print(
                        f"{method} accurate to a tolerance of {tol} for a system of ODEs with a step size of {step_size}")
                    break
            step_size *= step_size_multiplier

    """
    testing true against estimated values at t=1 for single ODE: dx/dt = a*x
    Increases step size until result is accurate to a specified tolerance,
    then returns the step size
    """
    """
    Euler method, tol = 1e-1
    """
    accuracy_test_single('euler', False, 1, 0.1, 1e-1, single_good_f, single_good_f_true, 1, 1)

    """
    Euler method, tol = 1e-3
    """
    accuracy_test_single('euler', False, 1, 0.1, 1e-3, single_good_f, single_good_f_true, 1, 1)

    """
    Euler method, tol = 1e-5
    """
    accuracy_test_single('euler', False, 1, 0.1, 1e-5, single_good_f, single_good_f_true, 1, 1)

    """
    4th order Runge-Kutta method, tol = 1e-1
    """
    accuracy_test_single('rk4', False, 1, 0.1, 1e-1, single_good_f, single_good_f_true, 1, 1)

    """
    4th order Runge-Kutta method, tol = 1e-3
    """
    accuracy_test_single('rk4', False, 1, 0.1, 1e-3, single_good_f, single_good_f_true, 1, 1)

    """
    4th order Runge-Kutta method, tol = 1e-5
    """
    accuracy_test_single('rk4', False, 1, 0.1, 1e-5, single_good_f, single_good_f_true, 1, 1)

    """
    testing true against estimated values at t=1 for a system of ODEs: dx/dt = a*y, dy/dt = b*x
    Increases step size until result is accurate to a specified tolerance,
    then returns the step size
    """
    """
    Euler method, tol = 1e-1
    """
    accuracy_test_single('euler', True, [1, 1], 0.1, 1e-1, system_good_f, system_good_f_true, 1, [1, -1])

    """
    Euler method, tol = 1e-3
    """
    accuracy_test_single('euler', True, [1, 1], 0.1, 1e-3, system_good_f, system_good_f_true, 1, [1, -1])

    """
    Euler method, tol = 1e-5
    """
    accuracy_test_single('euler', True, [1, 1], 0.1, 1e-5, system_good_f, system_good_f_true, 1, [1, -1])

    """
    4th order Runge-Kutta method, tol = 1e-1
    """
    accuracy_test_single('rk4', True, [1, 1], 0.1, 1e-1, system_good_f, system_good_f_true, 1, [1, -1])

    """
    4th order Runge-Kutta method, tol = 1e-3
    """
    accuracy_test_single('rk4', True, [1, 1], 0.1, 1e-3, system_good_f, system_good_f_true, 1, [1, -1])

    """
    4th order Runge-Kutta method, tol = 1e-5
    """
    accuracy_test_single('rk4', True, [1, 1], 0.1, 1e-5, system_good_f, system_good_f_true, 1, [1, -1])

    print("___________")
    print(" Value tests complete :)")
    print("___________")


def main():
    input_tests()
    value_tests()


if __name__ == '__main__':
    main()
