#! /usr/bin/python3

import numpy as np
import math
import matplotlib.pyplot as plt

T = np.linspace(0, 2 * math.pi, 1000)

X = [math.cos(t) for t in T]
Y = [-1 * math.sin(t) for t in T]
plt.plot(X, Y)
plt.axis('equal')
plt.title("Phase Plot for (x,y) = (cos(t), -sin(t), 0 <= t <= 2 pi")
plt.show()
