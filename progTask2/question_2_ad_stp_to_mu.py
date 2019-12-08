#! /usr/bin/python3

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import adaptive_rk as ad

global lmbd
global n
global y1
global y2
global mu
y1=2
y2=0
Mu = [1.5, 10, 15, 22, 33, 68, 100, 150, 220, 470, 680, 1000]

def f(t, xt):
    return np.array([xt[1], mu * (1 - xt[0] * xt[0]) * xt[1] - xt[0]])

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        y1 = float(sys.argv[1])
    if len(sys.argv) >= 3:
        y2 = float(sys.argv[2])

    step_count = []
    for mu in Mu:
        print("mu: " + str(mu))
        times, vals = ad.approximate(f=f,
                                     X0=np.array([y1, y2]),
                                     T=[0.0, 0.7 * mu],
                                     tau_max=0.5,
                                     tol=0.00001)
        step_count.append(len(times))
        print("Step count: " + str(step_count[-1]))

    #plt.loglog(Mu, step_count, label='Step Count')
    plt.plot(Mu, step_count, label='Step Count')
    #plt.axis('equal')
    plt.legend()

    plt.xlabel("Mu")
    plt.ylabel("Number of steps taken")
    plt.title("Step count to Mu for y1=" + str(y1) + \
            ", y2=" + str(y2) + \
            ", tol=" + str(0.00001))
    plt.show()
