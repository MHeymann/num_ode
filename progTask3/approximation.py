#! /usr/bin/python3

import numpy as np

def approximation(step, f, args=[], T=[0.0, 1.0], X0=np.array([1]), tau=0.1,):
    # set time limits
    t_init = T[0]
    t_final = T[1]

    # initialize estimator lists
    t_points = [t_init]
    X = [X0]

    # init loop variables
    t = t_init
    Z = X0

    while t + tau < t_final:
        # estimate
        Z = step(Z, t, t+tau, f, *args)

        # step time
        t = t + tau

        # add points
        t_points.append(t)
        X.append(Z)

    Z = step(Z, t, t_final, f, *args)
    X.append(Z)
    t_points.append(t_final)

    return (t_points, X)
