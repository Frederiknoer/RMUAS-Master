#!/usr/bin/env python
from shapely.geometry import Point, LineString, Polygon
import math
import numpy as np
import matplotlib.pyplot as plt

def define_triangle(a,b,c):
    angle_a = math.acos( (b**2 + c**2 - a**2) / (2 * b * c) )
    angle_b = math.acos( (a**2 + c**2 - b**2) / (2 * a * c) )
    angle_c = math.acos( (a**2 + b**2 - c**2) / (2 * a * b) )

    A = Point(0.0, 0.0)
    B = Point(c, 0.0)
    C = Point(b*math.cos(angle_a), b*math.sin(angle_a))

    return A, B, C, angle_a, angle_b, angle_c

def define_rectangle(a,b,c,d,e,f):
    A,B,C,angle_a1,angle_b1,_ = define_triangle(a,b,c)

    angle_a2 = math.acos( (b**2 + c**2 - a**2) / (2 * b * c) )
    angle_b2 = math.acos( (a**2 + c**2 - b**2) / (2 * a * c) )

    angle_c1 = math.acos( (b**2 + d**2 - e**2) / (2 * b * d) )
    angle_c2 = math.acos( (a**2 + d**2 - f**2) / (2 * a * d) )

    angle_d1 = math.acos( (e**2 + d**2 - b**2) / (2 * e * d) )
    angle_d2 = math.acos( (f**2 + d**2 - a**2) / (2 * f * d) )

    D = Point(b*math.cos(angle_a2), -b*math.sin(angle_a2))

    return A,B,C,D

def plot_points(x,y):
    plt.scatter(x,y)
    plt.show()


if __name__ == "__main__":
    range_a_b = 6.0
    range_a_c = 5.0
    range_a_d = 5.0
    range_b_c = 5.0
    range_b_d = 5.0
    range_c_d = 8.0

    A, B, C, D = define_rectangle(range_b_c, range_a_c, range_a_b, range_c_d, range_a_c, range_b_c)
    x = [A.x, B.x, C.x, D.x]
    y = [A.y, B.y, C.y, D.y]

    print("A: ", A.x, A.y, "  B: ", B.x, B.y, "  C: ", C.x, C.y, "  D: ", D.x, D.y)

    plot_points(x, y)
