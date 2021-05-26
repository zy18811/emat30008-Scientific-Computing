import orbit_shooting_tests
import solve_ode_tests
import continuation_tests

"""
runs all tests
"""


def main():
    print("Testing solve_ode():")
    solve_ode_tests.main()

    print("Testing orbit_shoooting():")
    orbit_shooting_tests.main()

    print("Testing continuation():")
    continuation_tests.main()


if __name__ == '__main__':
    main()