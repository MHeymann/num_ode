#! /usr/bin/python3

import numpy as np
import approximation as ap


A3 = np.array([
        [ 0.0, 0.0, 0.0],
        [ 0.5, 0.0, 0.0],
        [-1.0, 2.0, 0.0]
    ])
B3 = np.array([1.0/6, 2.0/3, 1.0/6])
C3 = np.array([
    0.0,
    1.0/2,
    1.0])

A4 = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ])
B4 = np.array([1.0/6, 1.0/3, 1.0/3, 1.0/6])
C4 = np.array([
    0.0,
    1.0/2,
    1.0/2,
    1.0])

_A = None
_B = None
_C = None
_n = None
_f = None

def f_test(t, xt):
    return -5 * xt

def runge_kutta_step(xj0, tj0, tj1):
    return _runge_kutta_step(xj0, tj0, tj1, _A, _B, _C, _f)

def _runge_kutta_step(xj0, tj0, tj1, A, B, C, f):
    tau = tj1 - tj0
    n = len(C)

    # init K list with k1 at position 0
    K = [f(tj0 + tau * C[0], xj0)]

    # calculate k2 ... kn, using indices 1 ... n-1
    for x in range(n-1):
        i = x + 1
        sumi = 0
        for j in range(i):
            sumi = sumi + A[i][j] * K[j]
        ki = f(tj0 + C[i] * tau, xj0 + tau * sumi)

        K.append(ki)

    # calculate the weighted sum of ki's
    sumk = 0
    for i in range(n):
        sumk = sumk + B[i] * K[i]

    # the actual step
    return xj0 + tau * sumk

def runge_kutta(A, B, C, f=f_test, X0=np.array([1]), T=[0.0, 1.0], tau=0.1):
    global _A
    global _B
    global _C
    global _n
    global _f

    _A = A
    _B = B
    _C = C
    _n = len(C)
    _f = f

    return ap.approximation(runge_kutta_step, T=T, X0=X0, tau=tau)
