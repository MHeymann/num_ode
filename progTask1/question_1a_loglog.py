#! /usr/bin/python3

import sys
import math
import matplotlib.pyplot as plt

import approximation as ap

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        ap.lmbd = float(sys.argv[1])
    else:
        ap.lmbd = 1
    if len(sys.argv) >= 3:
        local = False
    else:
        local = True
    Tau = [0.5, 0.25, 0.2, 0.125, 0.1, 0.05, 0.025, 0.0125 ,0.01, 0.001, 0.0001]

    diffs = []
    for t in Tau:
        times, vals = ap.approximation(ap.explicit_euler_step, tau=t)
        if local:
            diffs.append(math.exp(-ap.lmbd * times[1]) - vals[1])
        else:
            diffs.append(abs(math.exp(-ap.lmbd * times[-1]) - vals[-1]))

    plt.loglog(Tau, diffs)
    plt.xlabel("Interval size")
    if local:
        plt.ylabel("Local error")
        plt.title("Explicit Euler Local Error for lambda=" + str(ap.lmbd))
    else:
        plt.ylabel("Global error at t=1")
        plt.title("Explicit Euler Global Error for lambda=" + str(ap.lmbd))
    plt.show()
