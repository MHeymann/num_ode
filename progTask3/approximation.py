#! /usr/bin/python3

import numpy as np

def approximation(step, f, args=[], T=[0.0, 1.0], X0=np.array([1]), tau=0.1,):
    # set time limits
    t_init = T[0]
    t_final = T[1]

    # initialize estimator lists
    t_points = []
    X = [X0]

    # init loop variables
    t = t_init
    Z = X0

    while t < t_final:
        Z_temp = step(Z, t, t+tau, f, *args)

        t_points.append(t)
        X.append(Z)

        # step time
        t = t + tau

    t_points.append(t)
    X.append(Z)
    return (tl, X)
