from solve_ode import solve_ode


def continuation(ode, x0, par0, par_to_vary, step_size, maxSteps, discretisation = shooting, solver = solve_ode):
    