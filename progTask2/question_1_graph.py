#! /usr/bin/python3

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import runge_kutta as rk

global lmbd
global Tau
global n
lmbd = 5.0
Tau = 0.1
n=3

def f(t, xt):
    return -1 * lmbd * xt

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        lmbd = float(sys.argv[1])

    if len(sys.argv) >= 3:
        Tau = float(sys.argv[2])

    if len(sys.argv) >= 4:
        n = 4

    if n == 3:
        A = rk.A3
        B = rk.B3
        C = rk.C3
    else:
        A = rk.A4
        B = rk.B4
        C = rk.C4

    times, vals = rk.runge_kutta(A, B, C, f=f,
                                 X0=np.array([1]),
                                 T=[0.0, 1.0],
                                 tau=Tau)

    e = [math.exp(-1 * lmbd * t) for t in np.linspace(0,1,1000)]
    plt.plot(np.linspace(0,1,1000), e, label='Solution')
    plt.plot(times, vals, label='RK Approximation')
    plt.legend()

    plt.xlabel("Time t")
    plt.ylabel("Estimated Value")
    plt.title("Runge-Kutta Approximation for lambda=" + str(lmbd) + \
            ", tau=" + str(Tau) + \
            ", n=" + str(n))
    plt.show()
