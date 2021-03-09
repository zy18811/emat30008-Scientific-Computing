import sys
import math


def euler_step(f, x1, t1, h):
    x2 = x1 + h * f(t1, x1)
    t2 = t1 + h
    return t2, x2


def rk4_step(f, x1, t1, h):
    m1 = f(t1, x1)
    m2 = f(t1 + h / 2, x1 + (h / 2) * m1)
    m3 = f(t1 + h / 2, x1 + (h / 2) * m2)
    m4 = f(t1 + h, x1 + h * m3)
    x2 = x1 + (h / 6) * (m1 + 2 * m2 + 2 * m3 + m4)
    t2 = t1 + h
    return t2, x2


def solve_to(step,f,x1,t1,t2,hmax):
    while True:
        if (t1+hmax)>t2:
            t1,x1 = step(f,x1,t1,t2-t1)
            break
        t1,x1 = step(f,x1,t1,hmax)
    return x1


def solve_ode(f,x0,tArr,method,hmax):
    if method == "rk4":
        step = rk4_step
    elif method == "euler":
        step = euler_step
    else:
        sys.exit("Method: \"%s\" is not valid. Please select a valid method" % method)
    x_sol = [x0]
    for i in range(len(tArr)-1):
        xi = solve_to(step,f,x_sol[i],tArr[i],tArr[i+1],hmax)
        x_sol.append(xi)
    return x_sol
