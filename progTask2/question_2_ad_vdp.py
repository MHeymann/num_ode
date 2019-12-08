#! /usr/bin/python3

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import adaptive_rk as ad

global lmbd
global n
global y1
global y2
global mu
y1=2
y2=0
mu=1.5

def f(t, xt):
    return np.array([xt[1], mu * (1 - xt[0] * xt[0]) * xt[1] - xt[0]])

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        mu = float(sys.argv[1])

    if len(sys.argv) >= 3:
        y1 = float(sys.argv[2])
    if len(sys.argv) >= 4:
        y2 = float(sys.argv[3])

    times, vals = ad.approximate(f=f,
                                 X0=np.array([y1, y2]),
                                 T=[0.0, 8 * mu],
                                 tau_max=0.5,
                                 tol=0.00001)

    appx = list(list(zip(*vals))[0])
    appy = list(list(zip(*vals))[1])
    plt.plot(appx, appy, label='RK Approximation')
    plt.axis('equal')
    plt.legend()

    plt.xlabel("y1 estimate")
    plt.ylabel("y2 estimate")
    plt.title("Addaptive Approximation for y1=" + str(y1) + \
            ", y2=" + str(y2) + \
            ", tol=" + str(0.00001) + \
            ", mu=" + str(mu))
    plt.show()
