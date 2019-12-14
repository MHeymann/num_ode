#! /usr/bin/python3
import sys
import math
import numpy as np

import implicit_midpoint as im
import matplotlib.pyplot as plt

global k
k = 1.0

Tau = [0.5, 0.25, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001]

def func_h(t, xt):
    return np.array([xt[1], -k * xt[0]])

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        k = float(sys.argv[1])

    if len(sys.argv) >= 3:
        t_final = math.pi * 2 * float(sys.argv[2])
    else:
        t_final = math.pi * 2

    if len(sys.argv) >= 4:
        local = True
    else:
        local = False

    diffs = []

    for tau in Tau:
        if local:
            sol_val = [math.cos(math.sqrt(k) * tau), \
                    -1 * math.sin(math.sqrt(k) * tau)]
        else:
            sol_val = [math.cos(math.sqrt(k) * t_final), \
                    -1 * math.sin(math.sqrt(k) * t_final)]
        times, vals = im.approximation(func_h, \
                                       T=[0.0, t_final], \
                                       X0=np.array([1.0, 0]), \
                                       tau=tau)
        if local:
            app_val = vals[1]
        else:
            app_val = vals[-1]
        diffs.append(np.linalg.norm(app_val - sol_val))

    plt.loglog(Tau, diffs, label='Implicit midpoint Approximation')
    plt.grid()

    plt.xlabel("Stepsize")
    if local:
        plt.ylabel("Local Error")
        plt.title("Implicit Midpoint Approximation Local Error for k=" +
                str(k))
    else:
        plt.ylabel("Global Error")
        plt.title("Implicit Midpoint Approximation Global Error for k=" +
                str(k))
    plt.show()
