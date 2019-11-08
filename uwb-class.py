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
            #print('B: ', self.incidenceMatrix)
            #print('New Col: ', col)
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

    def run():
        pass


if __name__ == "__main__":
    A = uwb_agent(ID=0, pos=Point(0,0))
    B = uwb_agent(ID=1, pos=Point(3,4))
    C = uwb_agent(ID=2, pos=Point(1,5))

    A.handle_range_msg(Id=B.id, nb_pos=B.pos)
    A.handle_range_msg(Id=C.id, nb_pos=C.pos)
    A.handle_other_msg(Id1=B.id, Id2=C.id, range=B.get_distance(C.pos))

    print(A.pairs)

    print(A.incidenceMatrix)

    print(A.P)
