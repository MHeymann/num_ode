#! /usr/bin/python3
import sys
import math
import numpy as np

import implicit_gaus as ig
import matplotlib.pyplot as plt

global k
k = 1.0

def func_h(t, xt):
    return np.array([xt[1], -k * xt[0]])

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        Tau = float(sys.argv[1])
    else:
        Tau = 0.1

    if len(sys.argv) >= 3:
        k = float(sys.argv[2])

    if len(sys.argv) >= 4:
        t_final = math.pi * 2 * float(sys.argv[3])
    else:
        t_final = math.pi * 2

    times, vals = ig.approximation(func_h, \
                                   #T=[0.0, 2 * math.pi], \
                                   T=[0.0, t_final], \
                                   X0=np.array([1.0, 0]), \
                                   tau=Tau)

    appx = list(list(zip(*vals))[0])
    appy = list(list(zip(*vals))[1])
    #plt.plot(times, appx, label='Implicit Gauss Approximation')
    plt.plot(appx, appy, label='Implicit Gauss Approximation')

    n = round(t_final/0.01)
    sol_times = [t * 0.01 for t in range(n)]
    if not sol_times[-1] == t_final:
        sol_times.append(t_final)
    solx = [math.cos(math.sqrt(k) * t) for t in sol_times]
    soly = [-1 * math.sin(math.sqrt(k) * t) for t in sol_times]
    plt.plot(solx, soly, label='solution')
    plt.legend()

    plt.xlabel("x1 value")
    plt.ylabel("x2 value")
    plt.title("Implicit Gauss Approximation for tau=" + str(Tau) + \
              ", k=" + str(k))
    plt.axis("equal")
    plt.show()
