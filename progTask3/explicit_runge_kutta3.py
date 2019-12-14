#! /usr/bin/python3

import numpy as np
import explicit_runge_kutta as rk


_A = np.array([
    [ 0.0, 0.0, 0.0],
    [ 0.5, 0.0, 0.0],
    [-1.0, 2.0, 0.0]
    ])
_b = np.array([1.0/6, 2.0/3, 1.0/6])

def approximation(f, T=[0.0, 1.0], X0=np.array([1]), tau=0.1):
    return rk.approximation(f, \
                            T=T, \
                            X0=X0, \
                            tau=tau, \
                            A=_A, \
                            b=_b)
