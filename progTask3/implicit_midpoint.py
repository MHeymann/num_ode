#! /usr/bin/python3
import sys
import numpy as np
import math

import approximation as ap
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

global _A
global _b

#_A = np.array([
#    [0.25, 0.25 - (math.sqrt(3.0) / 6.0)],
#    [0.25 + (math.sqrt(3.0) / 6.0), 0.25]
#    ])
_A = np.array([
    [5.0/36.0, 2.0/9 - (math.sqrt(15.0) / 15.0), 5.0/36 - (math.sqrt(15.0)/30.0)],
    [5.0/36.0 + (math.sqrt(15.0) / 24.0), 2.0/9.0,  5.0/36 - (math.sqrt(15.0)/24.0)],
    [5.0/36.0 + (math.sqrt(15.0) / 30.0), 2.0/9.0 +  (math.sqrt(15.0)/15.0),
        5.0/36.0]
    ])
#_A = np.array([
#    [0., 0.0],
#    [0.5 , 0.5]
#    ])

#_b = [0.5, 0.5]
_b = [5.0/18.0, 4.0/9.0, 5.0/18.0]

tau = 0.1

def circ_f(t, xt):
    return np.array([xt[1], -xt[0]])

def F(z, *args):
    x   = args[0][0]
    t   = args[0][1]
    tau = args[0][2]
    f   = args[0][3]
    A   = args[0][4]
    c   = args[0][5]

    z = np.array(z).reshape((len(A), len(x)))
    fz = []
    for i in range(len(A)):
        s = np.zeros(len(x))
        for j in range(len(A)):
            s = s + A[i][j] * f(t + tau * c[j], x + z[j])
        fz.append(z[i] - tau * s)
    return np.array(fz).reshape(len(A) * len(x))

def implicit_midpoint_step(xj0, tj0, tj1, f, *args):
    A = args[0]
    b = args[1]
    c = args[2]
    F_args = [xj0, tj0, tj1 - tj0, f, A, c]


    z = fsolve(F, np.zeros((len(b), len(xj0))), F_args)
    z = np.array(z).reshape((len(A), len(xj0)))

    s = np.zeros(len(xj0))
    for i in range(len(z)):
        s = s + b[i] * f(tj0 + c[i] * (tj1 - tj0), xj0 + z[i])


    return xj0 + (tj1 - tj0) * s

def approximation(T=[0.0, 1.0], X0=np.array([1, 0]), tau=0.1, A=None, b=None):
    if A == None:
        A = _A
    if b == None:
        b = _b
    c = []
    for i in range(len(A)):
        s = 0
        for j in range(len(A)):
            s = s + A[i][j]
        c.append(s)

    args = [A, b, c]
    return ap.approximation(implicit_midpoint_step, \
                            circ_f, \
                            args,
                            T=T, \
                            X0=X0, \
                            tau=tau)

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        Tau = float(sys.argv[1])
    else:
        Tau = 2 * math.pi / 7

    #times, vals = approximation(T=[0.0, 2 * math.pi], tau=Tau, A=[[0.5]], b=[1])
    times, vals = approximation(T=[0.0, 2 * math.pi], tau=Tau)
    #X = [math.cos(t) for t in T]
    #Y = [-1 * math.sin(t) for t in T]
    appx = list(list(zip(*vals))[0])
    appy = list(list(zip(*vals))[1])
    #plt.plot(X, Y, label='Solution')
    plt.plot(appx, appy, label='Euler Approximation')
    #plt.legend()

    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Implicit Midpoint Approximation for tau=" + str(Tau))
    plt.axis("equal")
    plt.show()
