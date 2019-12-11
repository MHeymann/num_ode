#! /usr/bin/python3
import sys
import math
import numpy as np

import implicit_midpoint as im
import matplotlib.pyplot as plt

global lmbd
lmbd = 5

def func_e(t, xt):
    return -1 * lmbd * xt

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        Tau = float(sys.argv[1])
    else:
        Tau = 0.1
    if len(sys.argv) >= 3:
        lmbd = float(sys.argv[2])

    times, vals = im.approximation(func_e, \
                                   #T=[0.0, 2 * math.pi], \
                                   T=[0.0, 1.0], \
                                   X0=np.array([1.0]), \
                                   tau=Tau)
    #X = [math.cos(t) for t in T]
    #Y = [-1 * math.sin(t) for t in T]
    #appx = list(list(zip(*vals))[0])
    #appy = list(list(zip(*vals))[1])
    #plt.plot(X, Y, label='Solution')
    #plt.plot(appx, appy, label='Implicit Midpoint Approximation')

    plt.plot(times, vals, label='Implicit Midpoint Approximation')
    sol_times = [t * 0.01 for t in range(101)]
    sol = [math.exp(-lmbd * t) for t in sol_times]
    plt.plot(sol_times, sol, label='solution')
    plt.legend()

    plt.xlabel("Time t")
    plt.ylabel("Values")
    plt.title("Implicit Midpoint Approximation for tau=" + str(Tau) + \
              ", lambda=" + str(lmbd))
    #plt.axis("equal")
    plt.show()
