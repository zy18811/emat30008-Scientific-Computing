from shooting import orbit_shooting
import numpy as np
from scipy.optimize import fsolve
from periodfinderforcheck import manual_period_finder

"""
tests for the orbit_shooting function
"""


def input_tests():
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
    def good_ode_pc(u0,args):
        return good_ode(0,u0,args)[0]

    """
    ode returning an output of wrong type
    """
    def ode_wrong_output_type(t,u0,args):
        return "wrong type"

    """
    ode returning an output of wrong shape
    """

    def ode_wrong_output_shape(t, u0, args):
        return [u0,u0]

    """
    pc returning an output of the wrong type
    """
    def pc_wrong_output_type(u0,args):
        return "wrong type"

    """
    pc returning an output of the wrong shape (ie. not scalar)
    """
    def pc_wrong_output_shape(u0,args):
        return [1,2,3]

    all_tests_passed = True

    """
    good ode, pc, and u0 test
    Testing when a good ODE and u0 are provided - no errors should be raised
    """
    try:
        orbit_shooting(good_ode, u0, good_ode_pc, fsolve, [1,0.1,0.16])
        print("good ode, pc, and u0: test passed")
    except (TypeError,ValueError):
        all_tests_passed = False
        print("good ode, pc, and u0: test failed")

    """
    ode not a function test
    Testing when ode param is not a function - a TypeError should be raised
    """
    try:
        orbit_shooting("not a function", u0, good_ode_pc, fsolve, [1,0.1,0.16])
        all_tests_passed = False
        print("ode not a function test: failed")
    except TypeError:
        print("ode not a function test: passed")

    """
    ode wrong output type test
    Testing when ode param returns an output of the wrong type - a TypeError should be raised
    """
    try:
        orbit_shooting(ode_wrong_output_type, u0, good_ode_pc, fsolve, [1,0.1,0.16])
        all_tests_passed = False
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
        print("pc wrong output type: test failed")
    except TypeError:
        print("pc wrong output type: test passed")

    """
    pc wrong output shape test
    Testing when pc param returns an output of the wrong shape - a ValueError should be raised
    """
    try:
        orbit_shooting(good_ode, u0, pc_wrong_output_shape, fsolve, [1, 0.1, 0.16])
        all_tests_passed = False
        print("pc wrong output shape: test failed")
    except ValueError:
        print("pc wrong output shape: test passed")

    """
    incorrect u0 type test
    Testing when the u0 param is an incorrect type - a TypeError should be raised
    """
    try:
        orbit_shooting(good_ode, "wrong u0 type", good_ode_pc, fsolve, [1, 0.1, 0.16])
        all_tests_passed = False
        print("incorrect u0 type: test failed")
    except TypeError:
        print("incorrect u0 type: test passed")

    if all_tests_passed:
        print("___________")
        print("All input tests passed :)")
        print("___________")
    else:
        print("___________")
        print("Some input tests failed :(")
        print("___________")


def value_tests():

    def hopfNormal(t, u, args):
        beta = args[0]
        sigma = args[1]

        u1 = u[0]
        u2 = u[1]

        du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)
        return np.array([du1dt, du2dt])

    def pcHopfNormal(u0, args):
        p = hopfNormal(1, u0, args)[0]
        return p

    def hopfNormal_explicit(t,phase,args):
        beta = args[0]

        u1 = np.sqrt(beta)*np.cos(t+phase)
        u2 = np.sqrt(beta)*np.sin(t+phase)
        return np.array([u1,u2])

    u0_hopfNormal = np.array([1.4, 0, 6.3])

    orbit = orbit_shooting(hopfNormal,u0_hopfNormal,pcHopfNormal,fsolve,[1,-1])

    true = hopfNormal_explicit(0,orbit[-1],[1,-1])

    if np.allclose(true,orbit[:-1]):
        print("close")




def main():
    #input_tests()
    value_tests()

if __name__ == "__main__":
    main()