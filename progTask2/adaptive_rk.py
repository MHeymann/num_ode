#! /usr/bin/python3

import numpy as np
import runge_kutta as rk
from runge_kutta import _runge_kutta_step as rk_step

def f_test(t, xt):
    return -5 * xt

def rk_3step(xj0, tj0, tj1, f):
    return rk_step(xj0, tj0, tj1, rk.A3, rk.B3, rk.C3, f)

def rk_4step(xj0, tj0, tj1, f):
    return rk_step(xj0, tj0, tj1, rk.A4, rk.B4, rk.C4, f)

def approximate(T=[0.0, 1.0], X0=np.array([1.0]), f=f_test, tol=0.1,
        tau_max=0.2, safety_fac=0.9):
    if safety_fac > 1 or safety_fac <=0:
        print("Bad safety factor given.  using default of 0.9")
        safety_fac = 0.9
    # set time limits
    t_init = T[0]
    t_final = T[1]

    # initialize estimator lists
    t_points = [t_init]
    X = [X0]
    errs = [tol]

    # initialize loop variables
    t = t_init
    tau = tau_max
    Z = X0

    while t < t_final:
        tau = tau_max

        Z3 = rk_3step(Z, t, t+tau, f)
        Z4 = rk_4step(Z, t, t+tau, f)
        err = np.linalg.norm(Z3 - Z4)
        while err > tol:
            # Need to make stepsize smaller
            tau = tau * pow(safety_fac * tol / err, 1/4)

            # calculate new estimates
            Z3 = rk_3step(Z, t, t+tau, f)
            Z4 = rk_4step(Z, t, t+tau, f)
            err = np.linalg.norm(Z3 - Z4)
        #print(tau)

        # if we are still within time ranges, add to list
        if t + tau < t_final:
            t_points.append(t + tau)
            X.append(Z4)
            Z = Z4
            errs.append(err)

        # step time
        t = t + tau

    # add estimate for t_final
    Z4 = rk_4step(Z, t - tau, t_final, f)
    Z3 = rk_3step(Z, t - tau, t_final, f)
    err = np.linalg.norm(Z3 - Z4)

    t_points.append(t_final)
    X.append(Z4)
    errs.append(err)

    return (t_points, X)

