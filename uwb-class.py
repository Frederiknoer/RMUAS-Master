#!/usr/bin/env python
from shapely.geometry import Point, LineString, Polygon
import math
import numpy as np
import matplotlib.pyplot as plt

class uwb_agent:
    def __init__(self, ID, pos):
        self.id = ID
        self.pos = pos
        self.incidenceMatrix = np.array([])
        self.M = [self.id]
        self.N = []
        self.E = []
        self.P = []
        self.pairs = []
        self.I = np.array([[1,0],[0,1]])

    def get_distance(self, remote_pos):
        p1 = self.pos
        p2 = remote_pos
        dist = math.sqrt( (p2.x - p1.x)**2 + (p2.y - p1.y)**2 )
        return dist

    def update_incidenceMatrix(self):
        self.incidenceMatrix = np.array([])
        self.P = []
        rows = len(self.M)
        cols = len(self.pairs)
        self.incidenceMatrix = np.zeros((rows,cols), dtype=int)
        for i, pair in enumerate(self.pairs):
            col = np.zeros(rows, dtype=int)
            m1 = pair[0]
            m2 = pair[1]
            col[m1] = 1
            col[m2] = -1
            self.incidenceMatrix[:,i] = col.T
            self.P.append(pair[2])


    def add_nb_module(self, Id, range):
        if not any(x == Id for x in self.N):
            self.N.append(Id)
            self.E.append(range)
            self.M.append(Id)
            self.pairs.append([self.id, Id, range])
        else:
            self.E[Id] = range
            for pair in self.pairs:
                if any(x == Id for x in pair) and any(x == self.id for x in pair):
                    pair[2] = range

    def add_pair(self, Id1, Id2, range):
        pairs_present = 0
        for i, pair in enumerate(self.pairs):
            if (pair[0] == Id1 and pair[1] == Id2) or (pair[1] == Id1 and pair[0] == Id2):
                self.pairs[i][2] = range
            else:
                pairs_present += 1
        if pairs_present <= len(self.pairs):
            self.pairs.append([Id1, Id2, range])


    def handle_range_msg(self, Id, nb_pos):
        range = self.get_distance(nb_pos)
        self.add_nb_module(Id, range)
        self.update_incidenceMatrix()

    def handle_other_msg(self, Id1, Id2, range):
        self.add_pair(Id1, Id2, range)
        self.update_incidenceMatrix()

    def define_triangle(self,a,b,c):
        angle_a = math.acos( (b**2 + c**2 - a**2) / (2 * b * c) )
        angle_b = math.acos( (a**2 + c**2 - b**2) / (2 * a * c) )
        angle_c = math.acos( (a**2 + b**2 - c**2) / (2 * a * b) )

        A = Point(0.0, 0.0)
        B = Point(c, 0.0)
        C = Point(b*math.cos(angle_a), b*math.sin(angle_a))

        return A, B, C, angle_a, angle_b, angle_c

    def define_rectangle(self):
        a,b,c,d,e,f = self.P
        A,B,C,angle_a1,angle_b1,_ = self.define_triangle(a,b,c)

        angle_a2 = math.acos( (b**2 + c**2 - a**2) / (2 * b * c) )
        #angle_b2 = math.acos( (a**2 + c**2 - b**2) / (2 * a * c) )

        #angle_c1 = math.acos( (b**2 + d**2 - e**2) / (2 * b * d) )
        #angle_c2 = math.acos( (a**2 + d**2 - f**2) / (2 * a * d) )

        #angle_d1 = math.acos( (e**2 + d**2 - b**2) / (2 * e * d) )
        #angle_d2 = math.acos( (f**2 + d**2 - a**2) / (2 * f * d) )

        D = Point(b*math.cos(angle_a2), -b*math.sin(angle_a2))

        return A,B,C,D

    def run():
        pass


if __name__ == "__main__":
    A = uwb_agent(ID=0, pos=Point(0,0))
    B = uwb_agent(ID=1, pos=Point(3,4))
    C = uwb_agent(ID=2, pos=Point(1,5))
    D = uwb_agent(ID=3, pos=Point(6,2))

    A.handle_range_msg(Id=B.id, nb_pos=B.pos)
    A.handle_range_msg(Id=C.id, nb_pos=C.pos)
    A.handle_range_msg(Id=D.id, nb_pos=D.pos)

    A.handle_other_msg(Id1=B.id, Id2=C.id, range=B.get_distance(C.pos))
    A.handle_other_msg(Id1=B.id, Id2=D.id, range=B.get_distance(D.pos))
    A.handle_other_msg(Id1=C.id, Id2=D.id, range=C.get_distance(D.pos))

    p1, p2, p3, p4 = A.define_rectangle()
    x = [p1.x, p2.x, p3.x, p4.x]
    y = [p1.y, p2.y, p3.y, p4.y]

    print("A: ", p1.x, p1.y, "  B: ", p2.x, p2.y, "  C: ", p3.x, p3.y, "  D: ", p4.x, p4.y)
    print(A.pairs)
    print(A.incidenceMatrix)
    print(A.P)

    plt.scatter(x,y)
    plt.show()
