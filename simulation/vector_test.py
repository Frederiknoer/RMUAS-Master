from scipy import linalg as la
import matplotlib.pyplot as pl
import numpy as np
import math
import sys
import random

A = np.array([0.0,   0.0,   0.0])
B = np.array([3.0,   0.0,   0.0])
C = np.array([1.5, 2.59808, 0.0])

x = np.array([2.0,  1.5,  2.5])

z = np.array([  np.linalg.norm(x - A),
                np.linalg.norm(x - B),
                np.linalg.norm(x - C)   ])

p_mag1 = np.linalg.norm(x - A)
p_mag2 = np.linalg.norm(x - B)
p_mag3 = np.linalg.norm(x - C)

H = np.array([ [ (x[0] / p_mag1), (x[1] / p_mag1), (x[2] / p_mag1) ],  #])
               [ (x[0] / p_mag2), (x[1] / p_mag2), (x[2] / p_mag2) ], 
               [ (x[0] / p_mag3), (x[1] / p_mag3), (x[2] / p_mag3) ] ])

res = z - np.dot(H, x)
truth = z

print ("Result: ",  res)
print ("Truth: ", truth)