#! /usr/bin/python3
import sys
import numpy as np
import math

import implicit_runge_kutta as irk
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

global _A
global _b

_A=[[0.5]]
_b=[1]

tau = 0.1

def func(t, xt):
    return np.array([xt[1], -xt[0]])

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        Tau = float(sys.argv[1])
    else:
        Tau = 2 * math.pi / 100

    #times, vals = approximation(T=[0.0, 2 * math.pi], tau=Tau, A=[[0.5]], b=[1])
    times, vals = approximation(T=[0.0, 2 * math.pi], tau=Tau)
    #X = [math.cos(t) for t in T]
    #Y = [-1 * math.sin(t) for t in T]
    appx = list(list(zip(*vals))[0])
    appy = list(list(zip(*vals))[1])
    #plt.plot(X, Y, label='Solution')
    plt.plot(appx, appy, label='Euler Approximation')
    #plt.legend()

    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Implicit Midpoint Approximation for tau=" + str(Tau))
    plt.axis("equal")
    plt.show()
