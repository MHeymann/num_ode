#! /usr/bin/python3

import numpy as np

lmbd = 1.0

def f(t, xt):
    return -1 * lmbd * xt

def explicit_euler_step(xj0, tj0, tj1):
    return xj0 + (tj1 - tj0) * f(tj0, xj0)

def heuns_step(xj0, tj0, tj1):
    temp = (1 - lmbd * (tj1 - tj0))
    return 0.5 * (1 + temp * temp) * xj0

def implicit_step(x, tj0, tj1):
    tau = tj1 - tj0
    oneover = 1. / (1 + tau * tau)
    hover = tau * oneover
    x0 = oneover * x[0] + hover * x[1]
    x1 = oneover * x[1] - hover * x[0]
    return np.array([x0, x1])

def explicit_euler_q2_step(x, tj0, tj1):
    tau = tj1 - tj0
    return np.array([x[0] + tau * x[1], x[1] - tau * x[0]])

def approximation(step, T=[0.0, 1.0], X0=np.array([1]), tau=0.1):
    t0 = T[0]
    tf = T[1]
    tl = []
    X = []
    t = t0
    Z = X0

    while t < tf:
        tl.append(t)
        X.append(Z)
        Z = step(Z, t, t+tau)
        t = t + tau

    tl.append(t)
    X.append(Z)
    return (tl, X)
