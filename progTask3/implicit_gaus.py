#! /usr/bin/python3
import sys
import numpy as np
import math

import implicit_runge_kutta as irk
import matplotlib.pyplot as plt

global _A
global _b

_A = np.array([
    [0.25, 0.25 - (math.sqrt(3.0) / 6.0)],
    [0.25 + (math.sqrt(3.0) / 6.0), 0.25]
    ])

_b = [0.5, 0.5]

tau = 0.1

def func_circ(t, xt):
    return np.array([xt[1], -xt[0]])

def func_e(t, xt):
    return -5 * xt

def approximation(f, T, X0, tau):
    return irk.approximation(f, \
                             T=T, \
                             X0=X0, \
                             tau=tau, \
                             A=_A, \
                             b=_b)
