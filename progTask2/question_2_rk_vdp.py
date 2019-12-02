#! /usr/bin/python3

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import runge_kutta as rk

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

    A = rk.A4
    B = rk.B4
    C = rk.C4

    times, vals = rk.runge_kutta(A, B, C, f=f,
                                 X0=np.array([y1, y2]),
                                 T=[0.0, 4 * mu],
                                 tau=Tau)

    print(vals)
    appx = list(list(zip(*vals))[0])
    appy = list(list(zip(*vals))[1])
    plt.plot(appx, appy, label='RK Approximation')
    plt.axis('equal')
    plt.legend()

    plt.xlabel("Time t")
    plt.ylabel("Estimated Value")
    plt.title("Runge-Kutta Approximation for y1=" + str(y1) + \
            ", y2=" + str(y2) + \
            ", tau=" + str(Tau) + \
            ", mu=" + str(mu))
    plt.show()
