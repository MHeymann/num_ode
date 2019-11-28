#! /usr/bin/python3

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import approximation as ap

timeF = 2 * math.pi

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        Tau = float(sys.argv[1])
    else:
        Tau = 0.1
    if len(sys.argv) >= 3:
        timeF = float(sys.argv[2])

    times, vals = ap.approximation(ap.explicit_euler_q2_step, \
                                   tau=Tau, \
                                   T=[0, timeF], \
                                   X0=np.array([1.0, 0.0]))
    c = [np.array([math.cos(t), math.sin(t)]) for t in \
            np.linspace(0,2 * math.pi,1000)]

    solutionx = list(list(zip(*c))[0])
    solutiony = list(list(zip(*c))[1])
    plt.plot(solutionx, solutiony, label='Solution')

    appx = list(list(zip(*vals))[0])
    appy = list(list(zip(*vals))[1])

    plt.plot(appx, appy, label='Explicit Approximation')
    plt.legend()

    plt.xlabel("Time t")
    plt.ylabel("Estimated Value")
    plt.axis('equal')
    plt.title("Explicit Approximation for tau=" + str(Tau) +
            " with t between 0 and " + str(round(timeF, 2)) + ".")
    plt.show()
