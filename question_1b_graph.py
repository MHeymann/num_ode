#! /usr/bin/python3

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import approximation as ap

lmbd = 1.0

def f(t, xt):
    return -1 * lmbd * xt

def explicit_euler_step(xj0, tj0, tj1):
    return xj0 + (tj1 - tj0) * f(tj0, xj0)

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        lmbd = float(sys.argv[1])
    else:
        lmbd = 1.0

    if len(sys.argv) >= 3:
        Tau = float(sys.argv[2])
    else:
        Tau = 0.1

    times, vals = ap.approximation(explicit_euler_step, tau=Tau)
    e = [math.exp(-1 * lmbd * t) for t in np.linspace(0,1,1000)]
    plt.plot(np.linspace(0,1,1000), e, label='Solution')
    plt.plot(times, vals, label='Heuns Approximation')
    plt.legend()

    plt.xlabel("Time t")
    plt.ylabel("Estimated Value")
    plt.title("Explicit Heuns Approximation for lambda=" + str(lmbd) + \
            ", tau=" + str(Tau))
    plt.show()
