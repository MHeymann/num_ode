#! /usr/bin/python3

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import adaptive_rk as ad

global lmbd
global Tau
global n
global tol
global safety_fac

lmbd = 5.0
Tau = 0.5
n=3
tol = 0.01
safety_fac = 0.9

def f(t, xt):
    return -1 * lmbd * xt

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        lmbd = float(sys.argv[1])

    if len(sys.argv) >= 3:
        Tau = float(sys.argv[2])

    if len(sys.argv) >= 4:
        tol = float(sys.argv[3])

    if len(sys.argv) >= 5:
        safety_fac = float(sys.argv[4])

    times, vals = ad.approximate(T=[0.0, 1.0], X0=np.array([1.0]), f=f,
            tol=tol, tau_max=Tau, safety_fac=safety_fac)

    e = [math.exp(-1 * lmbd * t) for t in np.linspace(0,1,1000)]
    plt.plot(np.linspace(0,1,1000), e, label='Solution')
    plt.plot(times, vals, label='RK Approximation')
    plt.legend()

    plt.xlabel("Time t")
    plt.ylabel("Estimated Value")
    plt.title("Adaptive Approximation for lambda=" + str(lmbd) + \
            ", tau_max=" + str(Tau) + \
            ", rho=" + str(safety_fac))
    plt.show()
