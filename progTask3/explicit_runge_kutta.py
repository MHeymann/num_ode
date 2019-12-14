#! /usr/bin/python3

import numpy as np
import approximation as ap


A3 = np.array([
        [ 0.0, 0.0, 0.0],
        [ 0.5, 0.0, 0.0],
        [-1.0, 2.0, 0.0]
    ])
b3 = np.array([1.0/6, 2.0/3, 1.0/6])

_A = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
_b = np.array([1.0/6, 1.0/3, 1.0/3, 1.0/6])

def f_test(t, xt):
    return -5 * xt

def runge_kutta_step(xj0, tj0, tj1, f, *args):
    A = args[0]
    b = args[1]
    c = args[2]

    tau = tj1 - tj0
    n = len(c)

    # init K list with k1 at position 0
    K = [f(tj0 + tau * c[0], xj0)]

    # calculate k2 ... kn, using indices 1 ... n-1
    for x in range(n-1):
        i = x + 1
        sumi = 0
        for j in range(i):
            sumi = sumi + A[i][j] * K[j]
        ki = f(tj0 + c[i] * tau, xj0 + tau * sumi)

        K.append(ki)

    # calculate the weighted sum of ki's
    sumk = 0
    for i in range(n):
        sumk = sumk + b[i] * K[i]

    # the actual step
    return xj0 + tau * sumk

def approximation(f, T=[0.0, 1.0], X0=np.array([1]), tau=0.1, A=None, b=None):
    if A is None:
        A = _A
    if b is None:
        b = _b
    c = []
    for i in range(len(A)):
        s = 0
        for j in range(len(A)):
            s = s + A[i][j]
        c.append(s)

    args = [A, b, c]

    return ap.approximation(runge_kutta_step, \
                            f, \
                            args, \
                            T=T, \
                            X0=X0, \
                            tau=tau)
