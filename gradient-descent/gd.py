# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.pyplot as plt

def grad(x):
    return 2*(x-1)

def cost(x):
    return (x-1)**2

def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        # if abs(grad(x_new)) < 1e-3:
        #     break
        print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x[-1], cost(x[-1]), it))
        x.append(x_new)
    return (x, it)


(x1, it1) = myGD1(.2, 5)
# (x2, it2) = myGD1(.1, 5)
# print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
# print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))
