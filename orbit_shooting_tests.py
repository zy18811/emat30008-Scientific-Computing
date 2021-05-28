import numpy as np
from scipy.optimize import fsolve

from newtonrhapson import newton
from periodfinderforcheck import manual_period_finder
from shooting import orbit_shooting
from solve_ode import solve_ode

"""
tests for the orbit_shooting() function
"""


def input_tests():
    """
    Tests for whether orbit_shooting() can handle good/bad input parameters
    """
    """
    good u0 values
    """
    u0 = np.array([0.5, 0.5, 15])

    """
    good function
    """

    def good_ode(t, u0, args):
        x = u0[0]
        y = u0[1]

        a = args[0]
        d = args[1]
        b = args[2]

        dxdt = x * (1 - x) - (a * x * y) / (d + x)
        dydt = b * y * (1 - (y / x))
        return np.array([dxdt, dydt])

    """
    correct phase condition for good_f()
    """

    def good_ode_pc(u0, args):
        return good_ode(0, u0, args)[0]

    """
    ode returning an output of wrong type
    """

    def ode_wrong_output_type(t, u0, args):
        return "wrong type"

    """
    ode returning an output of wrong shape
    """

    def ode_wrong_output_shape(t, u0, args):
        return [u0, u0]

    """
    pc returning an output of the wrong type
    """

    def pc_wrong_output_type(u0, args):
        return "wrong type"

    """
    pc returning an output of the wrong shape (ie. not scalar)
    """

    def pc_wrong_output_shape(u0, args):
        return [1, 2, 3]

    all_tests_passed = True
    failed_tests = []

    """
    Tests Start
    """
    """
    good ode, pc, and u0 test
    Testing when a good ODE and u0 are provided - no errors should be raised
    """
    try:
        orbit_shooting(good_ode, u0, good_ode_pc, fsolve, [1, 0.1, 0.16])
        print("good ode, pc, and u0: test passed")
    except (TypeError, ValueError):
        all_tests_passed = False
        failed_tests.append("good ode, pc, and u0")
        print("good ode, pc, and u0: test failed")

    """
    ode not a function test
    Testing when ode param is not a function - a TypeError should be raised
    """
    try:
        orbit_shooting("not a function", u0, good_ode_pc, fsolve, [1, 0.1, 0.16])
        all_tests_passed = False
        failed_tests.append("ode not a function")
        print("ode not a function test: failed")
    except TypeError:
        print("ode not a function test: passed")

    """
    ode wrong output type test
    Testing when ode param returns an output of the wrong type - a TypeError should be raised
    """
    try:
        orbit_shooting(ode_wrong_output_type, u0, good_ode_pc, fsolve, [1, 0.1, 0.16])
        all_tests_passed = False
        failed_tests.append("ode wrong output type")
        print("ode wrong output type: test failed")
    except TypeError:
        print("ode wrong output type: test passed")

    """
    ode wrong output shape test
    Testing when ode param returns an output of the wrong shape - a ValueError should be raised
    """
    try:
        orbit_shooting(ode_wrong_output_shape, u0, good_ode_pc, fsolve, [1, 0.1, 0.16])
        all_tests_passed = False
        failed_tests.append("ode wrong output shape")
        print("ode wrong output shape: test failed")
    except ValueError:
        print("ode wrong output shape: test passed")

    """
    pc not a function test
    Testing when pc param is not a function - a TypeError should be raised
    """
    try:
        orbit_shooting(good_ode, u0, "not a function", fsolve, [1, 0.1, 0.16])
        all_tests_passed = False
        failed_tests.append("pc not a function")
        print("pc not a function test: failed")
    except TypeError:
        print("pc not a function test: passed")

    """
    pc wrong output type test
    Testing when pc param returns an output of the wrong type - a TypeError should be raised
    """
    try:
        orbit_shooting(good_ode, u0, pc_wrong_output_type, fsolve, [1, 0.1, 0.16])
        all_tests_passed = False
        failed_tests.append("pc wrong output type")
        print("pc wrong output type: test failed")
    except TypeError:
        print("pc wrong output type: test passed")

    """
    pc wrong output shape test
    Testing when pc param returns an output of the wrong shape - a TypeError should be raised
    """
    try:
        orbit_shooting(good_ode, u0, pc_wrong_output_shape, fsolve, [1, 0.1, 0.16])
        all_tests_passed = False
        failed_tests.append("pc wrong output shape")
        print("pc wrong output shape: test failed")
    except TypeError:
        print("pc wrong output shape: test passed")

    """
    incorrect u0 type test
    Testing when the u0 param is an incorrect type - a TypeError should be raised
    """
    try:
        orbit_shooting(good_ode, "wrong u0 type", good_ode_pc, fsolve, [1, 0.1, 0.16])
        all_tests_passed = False
        failed_tests.append("incorrect u0 type")
        print("incorrect u0 type: test failed")
    except TypeError:
        print("incorrect u0 type: test passed")

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
        [print(fail) for fail in failed_tests]


def value_tests():
    """
    Tests for whether orbit_shooting() produces correct output values
    """
    all_tests_passed = True
    failed_tests = []

    """
    Tests Start
    """
    """
    predator-prey good initial values test
    Testing using the predator-prey equations with good initial values provided
    Explicit solution is a numeric estimate, not an analytical solution so comparison between explicit and shooting 
    solutions are made at 3 significant figures.
    """

    """
    Function for predator-prey 
    """

    def predator_prey(t, u0, args):

        x = u0[0]
        y = u0[1]

        a = args[0]
        d = args[1]
        b = args[2]

        dxdt = x * (1 - x) - (a * x * y) / (d + x)
        dydt = b * y * (1 - (y / x))
        return np.array([dxdt, dydt])

    """
    Phase condition for predator_prey
    """

    def pc_predator_prey(u0, args):
        p = predator_prey(1, u0, args)[0]
        return p

    """
    Initial values close to the solution for predator-prey
    """
    good_predator_prey_u0 = np.array([0.07, 0.16, 23])

    """
    Getting true value of periodic orbit for predator-prey
    """
    t = np.linspace(0, 1000, 10000)
    predator_prey_solution = solve_ode(predator_prey, good_predator_prey_u0[:-1], t, "rk4", 0.001, True, [1, 0.1, 0.16])
    predator_prey_x = predator_prey_solution[0]
    predator_prey_y = predator_prey_solution[1]

    true_orbit = manual_period_finder(t, predator_prey_x, predator_prey_y)
    # rounds to 3 significant figures
    true_orbit = [np.float_("%.3g" % ele) for ele in true_orbit]

    """
    Using fsolve solver
    """
    shooting_orbit = orbit_shooting(predator_prey, good_predator_prey_u0, pc_predator_prey, fsolve, [1, 0.1, 0.16])
    shooting_orbit = [np.float_("%.3g" % ele) for ele in shooting_orbit]
    # checking if root finder did not converge
    if len(shooting_orbit) == 0:
        all_tests_passed = False
        failed_tests.append("predator-prey good initial values fsolve")
        print("predator-prey good initial values fsolve: test failed")

    # checking if solution found by shooting is close to explicit solution
    else:
        if np.allclose(true_orbit, shooting_orbit):
            print("predator-prey good initial values fsolve: test passed")
        else:
            all_tests_passed = False
            failed_tests.append("predator-prey good initial values fsolve")
            print("predator-prey good initial values fsolve: test failed")

    """
    Using newton solver
    """
    shooting_orbit = orbit_shooting(predator_prey, good_predator_prey_u0, pc_predator_prey, newton, [1, 0.1, 0.16])
    shooting_orbit = [np.float_("%.3g" % ele) for ele in shooting_orbit]
    # checking if root finder did not converge
    if len(shooting_orbit) == 0:
        all_tests_passed = False
        failed_tests.append("predator-prey good initial values newton")
        print("predator-prey good initial values newton: test failed")

    # checking if solution found by shooting is close to explicit solution
    else:
        if np.allclose(true_orbit, shooting_orbit):
            print("predator-prey good initial values newton: test passed")
        else:
            all_tests_passed = False
            failed_tests.append("predator-prey good initial values newton")
            print("predator-prey good initial values newton: test failed")

    """
    predator-prey divergent initial values test
    Testing using the predator-prey equations with initial values which cause the shooting to not converge.
    In this case, an empty array, [], should be returned.
    """
    divergent_predator_prey_u0 = np.array([55, 77, -100])

    """
    Using fsolve solver
    """
    orbit = orbit_shooting(predator_prey, divergent_predator_prey_u0, pc_predator_prey, fsolve, [1, 0.1, 0.16])
    if len(orbit) == 0:
        print("predator-prey divergent initial values fsolve: test passed")
    else:
        all_tests_passed = False
        failed_tests.append("predator-prey divergent initial values fsolve")
        print("predator-prey divergent initial values fsolve: test failed")

    """
    Using newton solver
    """
    orbit = orbit_shooting(predator_prey, divergent_predator_prey_u0, pc_predator_prey, newton, [1, 0.1, 0.16])
    if len(orbit) == 0:
        print("predator-prey divergent initial values newton: test passed")
    else:
        all_tests_passed = False
        failed_tests.append("predator-prey divergent initial values newton")
        print("predator-prey divergent initial values newton: test failed")

    """
    hopf normal good initial values test
    Testing using the Hopf normal form equations with good initial values provided
    """

    """
    Function for Hopf normal form
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
    Phase condition function for Hopf normal form
    """

    def pc_hopf_normal(u0, args):
        p = hopf_normal(1, u0, args)[0]
        return p

    """
    Explicit true solution for Hopf normal form
    """

    def hopf_normal_explicit(t, phase, args):
        beta = args[0]

        u1 = np.sqrt(beta) * np.cos(t + phase)
        u2 = np.sqrt(beta) * np.sin(t + phase)
        return np.array([u1, u2])

    """
    Initial values close to the solution for Hopf normal form 
    """
    good_u0_hopf_normal = np.array([1.4, 0, 6.3])

    """
    Using fsolve solver
    """
    orbit = orbit_shooting(hopf_normal, good_u0_hopf_normal, pc_hopf_normal, fsolve, [1, -1])

    # checking if root finder did not converge
    if len(orbit) == 0:
        all_tests_passed = False
        failed_tests.append("hopf normal good initial values fsolve")
        print("hopf normal good initial values fsolve: test failed")

    # checking if solution found by shooting is close to explicit solution
    else:
        shooting_u = orbit[:-1]
        T = orbit[-1]

        true_u = hopf_normal_explicit(0, T, [1, -1])

        if np.allclose(true_u, shooting_u):
            print("hopf normal good initial values fsolve: test passed")
        else:
            all_tests_passed = False
            failed_tests.append("hopf normal good initial values fsolve")
            print("hopf normal good initial values fsolve: test failed")

    """
    Using newton solver
    """
    orbit = orbit_shooting(hopf_normal, good_u0_hopf_normal, pc_hopf_normal, newton, [1, -1])

    # checking if root finder did not converge
    if len(orbit) == 0:
        all_tests_passed = False
        failed_tests.append("hopf normal good initial values newton")
        print("hopf normal good initial values newton: test failed")

    # checking if solution found by shooting is close to explicit solution
    else:
        shooting_u = orbit[:-1]
        T = orbit[-1]

        true_u = hopf_normal_explicit(0, T, [1, -1])

        if np.allclose(true_u, shooting_u):
            print("hopf normal good initial values newton: test passed")
        else:
            all_tests_passed = False
            failed_tests.append("hopf normal good initial values newton")
            print("hopf normal good initial values newton: test failed")

    """
    hopf normal divergent initial values test
    Testing using the Hopf normal form equations with initial values which cause the shooting to not converge.
    In this case, an empty array, [], should be returned.
    """
    divergent_u0_hopf_normal = np.array([-100, 234, -10])

    """
    Using fsolve solver
    """
    orbit = orbit_shooting(hopf_normal, divergent_u0_hopf_normal, pc_hopf_normal, fsolve, [1, -1])
    if len(orbit) == 0:
        print("hopf normal divergent initial values fsolve: test passed")
    else:
        all_tests_passed = False
        failed_tests.append("hopf normal divergent values fsolve")
        print("hopf normal divergent initial values fsolve: test failed")

    """
    Using newton solver
    """
    orbit = orbit_shooting(hopf_normal, divergent_u0_hopf_normal, pc_hopf_normal, newton, [1, -1])
    if len(orbit) == 0:
        print("hopf normal divergent initial values newton: test passed")
    else:
        all_tests_passed = False
        failed_tests.append("hopf normal divergent values newton")
        print("hopf normal divergent initial values newton: test failed")

    """
    hopf normal varied dimensions good initial values test
    Testing using the Hopf normal form equations with an extra dimension du3/dt = -u3 added and good initial values
    provided
    """

    """
    Function for Hopf normal form with added dimension
    """

    def hopf_normal_vary_dim(t, u, args):
        beta = args[0]
        sigma = args[1]

        u1 = u[0]
        u2 = u[1]
        u3 = u[2]

        du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)
        du3dt = -u3
        return np.array([du1dt, du2dt, du3dt])

    """
    Phase condition function for Hopf normal form with added dimension
    """

    def pc_hopf_normal_vary_dim(u0, args):
        p = hopf_normal_vary_dim(1, u0, args)[0]
        return p

    """
    Explicit true solution for Hopf normal form with added dimension
    """

    def hopf_normal_vary_dim_explicit(t, phase, args):
        beta = args[0]
        u1 = np.sqrt(beta) * np.cos(t + phase)
        u2 = np.sqrt(beta) * np.sin(t + phase)
        u3 = np.exp(-(t + phase))
        return np.array([u1, u2, u3])

    """
    Initial values close to the solution for Hopf normal form with added dimension
    """
    good_u0_hopf_normal_vary_dim = np.array([1.4, 0, 1, 6.3])

    """
    Using fsolve solver
    """
    orbit = orbit_shooting(hopf_normal_vary_dim, good_u0_hopf_normal_vary_dim, pc_hopf_normal_vary_dim, fsolve, [1, -1])

    # checking if root finder did not converge
    if len(orbit) == 0:
        all_tests_passed = False
        failed_tests.append("hopf normal added dimension good initial values fsolve")
        print("hopf normal added dimension good initial values fsolve: test failed")

    # checking if solution found by shooting is close to explicit solution
    else:
        shooting_u = orbit[:-1]
        T = orbit[-1]

        true_u = hopf_normal_vary_dim_explicit(10 * T, T, [1, -1])

        if np.allclose(true_u, shooting_u):
            print("hopf normal added dimension good initial values fsolve: test passed")
        else:
            all_tests_passed = False
            failed_tests.append("hopf normal added dimension good initial values fsolve")
            print("hopf normal added dimension good initial values fsolve: test failed")

    """
    Using newton solver
    """
    orbit = orbit_shooting(hopf_normal_vary_dim, good_u0_hopf_normal_vary_dim, pc_hopf_normal_vary_dim, newton, [1, -1])

    # checking if root finder did not converge
    if len(orbit) == 0:
        all_tests_passed = False
        failed_tests.append("hopf normal added dimension good initial values newton")
        print("hopf normal added dimension good initial values newton: test failed")

    # checking if solution found by shooting is close to explicit solution
    else:
        shooting_u = orbit[:-1]
        T = orbit[-1]

        true_u = hopf_normal_explicit(0, T, [1, -1])

        if np.allclose(true_u, shooting_u):
            print("hopf normal added dimension good initial values newton: test passed")
        else:
            all_tests_passed = False
            failed_tests.append("hopf normal added dimension good initial values newton")
            print("hopf normal added dimension good initial values newton: test failed")

    """
    hopf normal added dimension divergent initial values test
    Testing using the Hopf normal form equations with an added dimension du3/dt = -u3 with initial values which cause 
    the shooting to not converge. In this case, an empty array, [], should be returned.
    """
    divergent_u0_hopf_normal_vary_dim = np.array([-100, 234, 10, -10])

    """
    Using fsolve solver
    """
    orbit = orbit_shooting(hopf_normal_vary_dim, divergent_u0_hopf_normal_vary_dim, pc_hopf_normal_vary_dim, fsolve,
                           [1, -1])
    if len(orbit) == 0:
        print("hopf normal added dimension divergent initial values fsolve: test passed")
    else:
        all_tests_passed = False
        failed_tests.append("hopf normal added dimension divergent values fsolve")
        print("hopf normal added dimension divergent initial values fsolve: test failed")

    """
    Using newton solver
    """
    orbit = orbit_shooting(hopf_normal_vary_dim, divergent_u0_hopf_normal_vary_dim, pc_hopf_normal_vary_dim, newton,
                           [1, -1])
    if len(orbit) == 0:
        print("hopf normal added dimension divergent initial values newton: test passed")
    else:
        all_tests_passed = False
        failed_tests.append("hopf normal added dimension divergent values newton")
        print("hopf normal added dimension divergent initial values newton: test failed")

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
        [print(fail) for fail in failed_tests]


def main():
    print("Input Tests:")
    input_tests()
    print("Value Tests:")
    value_tests()


if __name__ == "__main__":
    main()
