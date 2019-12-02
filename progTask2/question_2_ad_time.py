#! /usr/bin/python3

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import adaptive_rk as ad

global lmbd
global Tau
global n
global y1
global y2
global mu
Tau = 0.1
y1=2
y2=0
mu=1.5

def f(t, xt):
    return np.array([xt[1], mu * (1 - xt[0] * xt[0]) * xt[1] - xt[0]])

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        mu = float(sys.argv[1])

    if len(sys.argv) >= 3:
        Tau = float(sys.argv[2])

    if len(sys.argv) >= 4:
        y1 = float(sys.argv[3])
    if len(sys.argv) >= 5:
        y2 = float(sys.argv[4])

    times, vals = ad.approximate(f=f,
                                 X0=np.array([y1, y2]),
                                 T=[0.0, 1.97 * mu],
                                 tau_max=0.5,
                                 tol=0.00001)

    diffs = []
    for i in range(len(times) - 1):
        diffs.append(times[i + 1] - times[i])
    plt.plot(range(len(times) - 1), diffs, label='RK Approximation')
    plt.legend()

    plt.xlabel("Step i")
    plt.ylabel("Step size")
    plt.title("Addaptive Approximation time steps for y1=" + str(y1) + \
            ", y2=" + str(y2) + \
            ", tol=" + str(0.00001) + \
            ", mu=" + str(mu))
    plt.show()
