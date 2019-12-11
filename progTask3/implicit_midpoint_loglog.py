#! /usr/bin/python3
import sys
import math
import numpy as np

import implicit_midpoint as im
import matplotlib.pyplot as plt

global lmbd
global Tau
lmbd = 5
Tau = [0.9, 0.5, 0.25, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
#Tau = [0.9, 0.5, 0.25, 0.2, 0.1, 0.05, 0.01, 0.005]

def func_e(t, xt):
    return -1 * lmbd * xt

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        lmbd = float(sys.argv[1])

    if len(sys.argv) >= 3:
        local=False
    else:
        local=True

    diffs = []
    for tau in Tau:
        times, vals = im.approximation(func_e, \
                                       T=[0.0, 1.0], \
                                       X0=np.array([1.0]), \
                                       tau=tau)
        if not local:
            t = times[-1]
            diff = abs(math.exp(-lmbd * t) - vals[-1])
        else:
            t = times[1]
            diff = abs(math.exp(-lmbd * t) - vals[1])
        diffs.append(diff)


    plt.loglog(Tau, diffs, label='Error')

    plt.xlabel("Stepvalue size")
    if local:
        plt.ylabel("Local Error")
        plt.title("Implicit Midpoint Local Error for lambda=" + str(lmbd))
    else:
        plt.ylabel("Global Error")
        plt.title("Implicit Midpoint Global Error for lambda=" + str(lmbd))
    plt.grid()
    plt.show()
