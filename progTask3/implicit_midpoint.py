#! /usr/bin/python3
import sys
import numpy as np

import implicit_runge_kutta as irk
import matplotlib.pyplot as plt

global _A
global _b

_A = np.array([[0.5]])

_b = [1.0]

def approximation(f, T, X0, tau):
    return irk.approximation(f, \
                             T=T, \
                             X0=X0, \
                             tau=tau, \
                             A=_A, \
                             b=_b)
