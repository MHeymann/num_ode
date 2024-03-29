#! /usr/bin/python3

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import approximation as ap

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        ap.lmbd = float(sys.argv[1])
    else:
        ap.lmbd = 1.0

    if len(sys.argv) >= 3:
        Tau = float(sys.argv[2])
    else:
        Tau = 0.1

    if len(sys.argv) >= 4:
        t_final = float(sys.argv[3])
    else:
        t_final = 1.0

    times, vals = ap.approximation(ap.explicit_euler_step, \
                                   T=[0.0, t_final], \
                                   tau=Tau)

    n = round(t_final/0.001)
    sol_times = [0.001 * t for t in range(n)]
    if not sol_times[-1] == t_final:
        sol_times.append(t_final)
    e = [math.exp(-1 * ap.lmbd * t) for t in sol_times]
    plt.plot(sol_times, e, label='Solution')
    plt.plot(times, vals, label='Euler Approximation')
    plt.legend()

    plt.xlabel("Time t")
    plt.ylabel("Estimated Value")
    plt.title("Explicit Euler Approximation for lambda=" + str(ap.lmbd) + \
            ", tau=" + str(Tau))
    plt.show()
