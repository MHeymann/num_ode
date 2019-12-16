#! /usr/bin/python3
import sys
import math
import numpy as np

import implicit_midpoint as im
import matplotlib.pyplot as plt

global k
k = 1.0

def lambda_f(y1, y2):
    return k * (math.sqrt(y1 * y1 + y2 * y2) - 1) / (math.sqrt(y1 * y1 + y2 * y2))

def func_spring(t, xt):
    return np.array([
                     xt[2], \
                     xt[3], \
                     -1 * xt[0] * lambda_f(xt[0], xt[1]), \
                     -1 * xt[1] * lambda_f(xt[0], xt[1]) - 1
                     ])

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        Tau = float(sys.argv[1])
    else:
        Tau = 0.1

    if len(sys.argv) >= 3:
        k = float(sys.argv[2])

    if len(sys.argv) >= 4:
        x0 = float(sys.argv[3])
    else:
        x0 = 1.1

    if len(sys.argv) >= 5:
        y0 = float(sys.argv[4])
    else:
        y0 = 0

    if len(sys.argv) >= 6:
        t_final = math.pi * 2 * float(sys.argv[5])
    else:
        t_final = math.pi * 2 * 3

    times, vals = im.approximation(func_spring, \
                                   T=[0.0, t_final], \
                                   X0=np.array([x0, y0, 0.0, 0.0]), \
                                   tau=Tau)

    appx = list(list(zip(*vals))[0])
    appy = list(list(zip(*vals))[1])
    #plt.plot(times, appx, label='Implicit midpoint Approximation')
    plt.plot(appx, appy, label='Implicit Midpoint Approximation')

    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Implicit Midpoint Approximation for tau=" + str(Tau) + \
              ", k=" + str(k) + \
              ", (x0, y0)=(" + str(x0) + ", " + str(y0) + ")")
    plt.axis("equal")
    plt.show()
