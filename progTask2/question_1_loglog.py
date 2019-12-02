#! /usr/bin/python3

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import runge_kutta as rk

global lmbd
global n
lmbd = 5.0
n=3

def f(t, xt):
    return -1 * lmbd * xt

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        lmbd = float(sys.argv[1])

    if len(sys.argv) >= 3:
        n = int(sys.argv[2])

    if len(sys.argv) >= 4:
        local = False
    else:
        local = True

    Tau = [0.25, 0.2, 0.125, 0.1, 0.05, 0.025, 0.0125 ,0.01, 0.001]

    if n == 3:
        A = rk.A3
        B = rk.B3
        C = rk.C3
    else:
        A = rk.A4
        B = rk.B4
        C = rk.C4

    diffs = []
    for t in Tau:
        times, vals = rk.runge_kutta(A, B, C, f=f,
                                     X0=np.array([1]),
                                     T=[0.0, 1.0],
                                     tau=t)
        if local:
            diffs.append(abs(math.exp(-1 * lmbd * times[1]) - vals[1]))
        else:
            diffs.append(abs(math.exp(-1 * lmbd * times[-1]) - vals[-1]))

    plt.loglog(Tau, diffs)
    plt.xlabel("Interval size")
    if local:
        plt.ylabel("Local error")
        plt.title("Runge-Kutta Local Error for lambda=" + str(lmbd) + \
                ", n=" + str(n))
    else:
        plt.ylabel("Global error at t=1")
        plt.title("Runge-Kutta Global Error for lambda=" + str(lmbd) + \
                ", n=" + str(n))
    plt.grid()
    plt.show()
