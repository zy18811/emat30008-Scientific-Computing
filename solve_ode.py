import sys
import math


def euler_step(f, x1, t1, h):
    x2 = x1 + h * f(t1, x1)
    t2 = t1 + h
    return x2, t2


def rk4_step(f, x1, t1, h):
    m1 = f(t1, x1)
    m2 = f(t1 + h / 2, x1 + (h / 2) * m1)
    m3 = f(t1 + h / 2, x1 + (h / 2) * m2)
    m4 = f(t1 + h, x1 + h * m3)
    x2 = x1 + (h / 6) * (m1 + 2 * m2 + 2 * m3 + m4)
    t2 = t1 + h
    return x2, t2


def solve_to(step,f,x1,t1,t2,hmax):
    while (t1+hmax)<t2:
        x1,t1 = step(f,x1,t1,hmax)
    else:
        x1,t1 = step(f,x1,t1,t2-t1)
    return x1




def solve_ode(f,x0,tArr,method,hmax):
    if method == "euler":
        method = euler_step
    elif method == "rk4":
        method = rk4_step
    else:


def func(t,x):
    dxdt = x
    return dxdt


x = solve_to(euler_step,func,1,0,1,0.01)
print(x)