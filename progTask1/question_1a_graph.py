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

    times, vals = ap.approximation(ap.explicit_euler_step, tau=Tau)
    e = [math.exp(-1 * ap.lmbd * t) for t in np.linspace(0,1,1000)]
    plt.plot(np.linspace(0,1,1000), e, label='Solution')
    plt.plot(times, vals, label='Euler Approximation')
    plt.legend()

    plt.xlabel("Time t")
    plt.ylabel("Estimated Value")
    plt.title("Explicit Euler Approximation for lambda=" + str(ap.lmbd) + \
            ", tau=" + str(Tau))
    plt.show()
