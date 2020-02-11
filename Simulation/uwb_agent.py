#!/usr/bin/env python
'''
Position Rules:
ID = 0: Origin of the local coordinate system (a)
ID = 1: Always 0 in y- and z-components (b)
ID = 2: Always posetive in both x- and y-components
ID = 3: Always posetive in x-component and negative in y-component
ID = 10: Always the UAV
'''

import math
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import sympy as symp
import scipy as scip
import serial


PI = 3.14159265359
DEBUG = False

class KF:
    def __init__(self):
        self.dt = 5e-2
        self.x = np.array([ 1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0, ])



        self.P = np.eye(6)
        self.P *= 100.

        self.R = 1.0

        self.Q = np.eye(6)
        self.Q *= 0.01

        p_mag = np.linalg.norm(self.x[0:3])
        self.H = np.array([[ (self.x[0] / p_mag), (self.x[1] / p_mag), (self.x[2] / p_mag), 0, 0, 0 ]])

        self.F = np.array([ [1.0, 0.0, 0.0, self.dt, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, self.dt, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, self.dt],
                            [0.0, 0.0, 0.0, 1.0,  0.0,    0.0],
                            [0.0, 0.0, 0.0, 0.0,  1.0,    0.0],
                            [0.0, 0.0, 0.0, 0.0,  0.0,    1.0] ])


        self.G = np.array([ [0.5*self.dt**2, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.5*self.dt**2, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.5*self.dt**2, 0.0, 0.0, 0.0],
                            [self.dt,0.0 ,        0.0, 0.0, 0.0, 0.0],
                            [0.0,   self.dt,      0.0, 0.0, 0.0, 0.0],
                            [0.0,    0.0,     self.dt, 0.0, 0.0, 0.0] ])

        self.K = np.array([ 0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0, ])
        self.I = np.eye(6)

    def predict(self, acc_in):
        acc_in = np.append(acc_in, np.array([0,0,0]))
        self.x = np.matmul(self.F, self.x) + np.matmul(self.G, acc_in)
        self.P = np.matmul( self.F, np.matmul(self.P, self.F.transpose()) ) + np.matmul( self.G, np.matmul(self.P, self.G.transpose()) ) + self.Q

        '''
        self.pos = np.add(self.pos, np.add((self.v * self.dt), (np.matmul(self.G, acc_in))).reshape((3,1)))
        self.v = self.v + (self.dt * acc_in)

        self.P = np.matmul(np.matmul(self.F.transpose(), self.P), self.F) + np.matmul(np.matmul(self.G.transpose(), self.P), self.G)
        '''

    def update(self, range_in):
        err = abs(np.linalg.norm(self.x[0:3]) - range_in)

        k1 = np.matmul(self.P, self.H.transpose())
        hp = np.matmul(self.H, self.P)
        k2 = np.reciprocal(np.matmul(hp, self.H.transpose())  + self.R)

        self.K = k1 * k2

        self.pos = self.K * err
        self.P = np.multiply((self.I - np.multiply(self.K, self.H)), self.P)
        p_mag = np.linalg.norm(self.x[0:3])
        self.H = np.array([[ (self.x[0] / p_mag), (self.x[1] / p_mag), (self.x[2] / p_mag), 0, 0, 0 ]])
        #self.H = (np.array([[ (self.pos[0] / np.linalg.norm(self.pos)), (self.pos[1] / np.linalg.norm(self.pos)), (self.pos[2] / np.linalg.norm(self.pos)) ]])).reshape(1,3)


class uwb_agent:
    def __init__(self, ID):
        self.id = ID
        self.incidenceMatrix = np.array([])
        self.M = np.array([self.id]) #Modules
        self.N = np.array([]) #Neigbours
        self.E = np.array([0]) #
        self.P = np.array([]) #
        self.pairs = np.empty((0,3))
        self.errorMatrix = np.array([])
        self.des_dist = 15
        self.I = np.array([[1,0],[0,1]])
        self.poslist = np.array([])
        #self.ID_list = np.array(["FFFFE3BC", "FFFFA858", "6F0B"])
        self.KF = KF()

    def get_B(self):
        return self.incidenceMatrix

    def calc_dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def update_incidenceMatrix(self):
        self_counter = 0
        self.incidenceMatrix = np.array([])
        self.P = np.array([])
        self.incidenceMatrix = np.zeros((4,4), dtype=int)
        for i, pair in enumerate(self.pairs):
            if pair[0] == self.id or pair[1] == self.id:
                self_counter += 1
                continue
            else:
                col = np.zeros(3, dtype=int)
                m1 = int(pair[0])
                m2 = int(pair[1])
                col[m1] = 1
                col[m2] = -1
                self.incidenceMatrix[:,(i-self_counter)] = col.T
                self.P = np.append(self.P, pair[2])


    def add_nb_module(self, Id, range):
        if not any(x == Id for x in self.N):
            self.N = np.append(self.N, Id)
            self.E = np.append(self.E, range)
            self.M = np.append(self.M, Id)
            self.pairs = np.append(self.pairs, np.array([[self.id, Id, range]]),axis=0)
        else:
            self.E[Id] = range
            for pair in self.pairs:
                if any(x == Id for x in pair) and any(x == self.id for x in pair):
                    pair[2] = range

    def add_pair(self, Id1, Id2, range):
        pair_present = False
        for i, pair in enumerate(self.pairs):
            if (pair[0] == Id1 and pair[1] == Id2) or (pair[1] == Id1 and pair[0] == Id2):
                self.pairs[i][2] = range
                pair_present = True

        if not pair_present:
            self.pairs = np.append(self.pairs, np.array([[Id1, Id2, range]]),axis=0)


    def handle_range_msg(self, Id, range):
        self.add_nb_module(Id, range)
        self.KF.update(range)
        #self.update_incidenceMatrix()

    def handle_other_msg(self, Id1, Id2, range):
        self.add_pair(Id1, Id2, range)
        #self.update_incidenceMatrix()

    def handle_acc_msg(self, acc_in):
        self.KF.predict(acc_in)
        return self.KF.x

    def clean_cos(self, cos_angle):
        return min(1,max(cos_angle,-1))

    def calc_spheres(self, sphere_list):
        #Sphere = x_pos, y_pos, z_pos, radius
        #Sphere eq: (x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2 = r^2
        x,y,z = sp.symbols('x y z')
        eq_list = np.array([])

        for i, sph in enumerate(sphere_list):
            eq_list = np.append(eq_list, sp.Eq( (x - sph[0])**2 + (y - sph[1])**2 + (z - sph[2])**2, sph[3]**2 ) )

        result = sp.solve(eq_list, (x,y,z))
        return result


    def mse(self, x, locations, distances):
        mse = 0.0
        for location, distance in zip(locations, distances):
            distance_calculated = self.calc_dist(x,location)
            mse += math.pow(distance_calculated - distance, 2.0)
        return mse / len(distances)

    def calc_pos_MSE(self):
        locations = self.predefine_ground_plane()
        distances = [None]*4
        for i, pair in enumerate(self.pairs):
            if pair[0] == 10:
                if pair[1] == 0:
                    distances[0] = pair[2]
                elif pair[1] == 1:
                    distances[1] = pair[2]
                elif pair[1] == 2:
                    distances[2] = pair[2]
                elif pair[1] == 3:
                    distances[3] = pair[2]

        initial_location = np.array([-1.0, 3.5, 2.1])

        result = scip.optimize.minimize(
            self.mse,                    # The error function
            initial_location,            # The initial guess
            args=(locations, distances), # Additional parameters for mse
            method='L-BFGS-B',           # The optimisation algorithm
            options={
            'ftol':1e-5,                 # Tolerance
            'maxiter': 1e+7              # Maximum iterations
            })
        location = result.x

        return location

    def predefine_ground_plane(self):
        A = np.array([0.0, 0.0, 0.0])
        B = np.array([1.75, 0.0, 0.0])
        C = np.array([0.9, 1.55, 0.0])
        #D = np.array([2.0, 0.0, 0.0])

        self.poslist = np.array([A,B,C])
        return A, B, C#, D


    def define_ground_plane(self):
        '''
        Ground plane setup:
           C
        A     B
           D
        '''
        temp_pair = []
        #print(self.pairs)
        for i in range((self.pairs.shape[0] - 1)):
            temp_pair = [self.pairs[i][0], self.pairs[i][1], self.pairs[i][2]]

            if 0 in temp_pair[0:2]:
                idx1 = temp_pair[0:2].index(0)
                if temp_pair[abs(idx1-1)] == 1: #Takes the other element than idx1
                    c = temp_pair[2] #Range between ID0 and ID1
                elif temp_pair[abs(idx1-1)] == 2:
                    b = temp_pair[2] #Range between ID0 and ID2
                elif temp_pair[abs(idx1-1)] == 3:
                    e = temp_pair[2]

            elif 1 in temp_pair[0:2]:
                idx1 = temp_pair[0:2].index(1)
                if temp_pair[abs(idx1-1)] == 2:
                    a = temp_pair[2]
                elif temp_pair[abs(idx1-1)] == 3:
                    f = temp_pair[2]

            else:
                d = temp_pair[2]

        range_list = np.array([a,b,c,d,e,f])
        angle_a1 = math.acos(self.clean_cos( (b**2 + c**2 - a**2) / (2 * b * c) ))
        angle_a2 = math.acos(self.clean_cos( (e**2 + c**2 - f**2) / (2 * e * c) ))
        angle_b = math.acos(self.clean_cos( (a**2 + c**2 - b**2) / (2 * a * c) ))
        angle_c = math.acos(self.clean_cos( (a**2 + b**2 - c**2) / (2 * a * b) ))

        A = np.array([0.0, 0.0, 0.0])
        B = np.array([c, 0.0, 0.0])
        C = np.array([b*math.cos(angle_a1), b*math.sin(angle_a1), 0.0])
        D = np.array([e*math.cos(angle_a2), -e*math.sin(angle_a2), 0.0])

        self.poslist = np.array([A,B,C,D])
        return A, B, C, D

    def define_h_triangle(self):
        for i in range((self.pairs.shape[0] - 1)):
            pair = [self.pairs[i][0], self.pairs[i][1]]
            if self.id in pair:
                idx = pair.index(self.id)
                if temp_pair[abs(idx1-1)] == 0:
                    v_a = self.pairs[i][2]
                if temp_pair[abs(idx1-1)] == 1:
                    v_b = self.pairs[i][2]
                if temp_pair[abs(idx1-1)] == 2:
                    v_c = self.pairs[i][2]
                if temp_pair[abs(idx1-1)] == 3:
                    v_d = self.pairs[i][2]

        v_range_list = np.array([v_a, v_b, v_c, v_d])
        a,b,c,d,e,f,d = self.range_list
        '''
          UAV

        A     B
        '''
        angle_v_a = math.acos(self.clean_cos( (v_a**2 + c**2 - v_b**2) / (2 * v_a * c) ))
        UAV_alt = v_a*math.sin(angle_v_a)





    def define_triangle(self):
        range_list = np.array([])
        for pair in self.pairs:
            print pair
            if pair[0] == self.id or pair[1] == self.id:
                continue
            else:
                np.append(range_list, [pair[2]])
        c,b,a = range_list

        angle_a = math.acos(self.clean_cos( (b**2 + c**2 - a**2) / (2 * b * c) ))
        angle_b = math.acos(self.clean_cos( (a**2 + c**2 - b**2) / (2 * a * c) ))
        angle_c = math.acos(self.clean_cos( (a**2 + b**2 - c**2) / (2 * a * b) ))

        A = np.array([0.0, 0.0, 0.0])
        B = np.array([c, 0.0, 0.0])
        C = np.array([b*math.cos(angle_a), b*math.sin(angle_a), 0.0])

        self.poslist = A,B,C
        return A, B, C #, angle_a, angle_b, angle_c

    def calcErrorMatrix(self):
        arrSize = self.M.size

        self.errorMatrix = np.zeros((arrSize, arrSize))

        poslist = self.poslist
        print("ID: ",self.id,"  Poslist: ",poslist)

        for i in range(arrSize):
            for j in range(arrSize):
                curDis = self.calc_dist(poslist[i], poslist[j])
                if curDis == 0:
                    self.errorMatrix[i][j] = 0.0
                else:
                    self.errorMatrix[i][j] = curDis - self.des_dist
        print("Error Matrix: ")
        print(self.errorMatrix)

    def calc_u_acc(self):
        self.calcErrorMatrix()

        U = np.array([])
        K = 0.01
        E = self.errorMatrix

        for i in range(self.M.size):
            u_x = 0
            u_y = 0
            for k in range(self.M.size):
                if k != i:
                    x_dif = self.poslist[i][0] - self.poslist[k][0]
                    y_dif = self.poslist[i][1] - self.poslist[k][1]
                    xy_mag = self.calc_dist(self.poslist[i],self.poslist[k])
                    unitvec = (self.poslist[i] - self.poslist[k]) / xy_mag

                    u_x += K * E[i][k] * unitvec[0]
                    u_y += K * E[i][k] * unitvec[1]

            U = np.append(U, [u_x, u_y])
        print("U: ")
        print(U)
        return U


    def run():
        pass


'''
def get_uwb_id(data):
    id_idx = data.find('from:')
    if id_idx == -1:
        return None
    id = ''
    while data[id_idx+6] != ' ':
        id += data[id_idx+6]
        id_idx += 1
    id = id[:-1]
    print("Id raw: ", id)
    if id == "FFFFB7B1":
        return 0
    elif id == "FFFFE3BC":
        return 1
    elif id == "6F0B":
        return 2
    return id

def get_uwb_range(data):
    range_idx = data.find('Range:')
    if range_idx == -1:
        return None
    data_val = (float(data[range_idx+7])) + (float(data[range_idx+9])*0.1) + (float(data[range_idx+10])*0.01)
    return data_val

if __name__ == "__main__":
    ser = serial.Serial('/dev/ttyUSB0', 115200)
    UAV_agent = uwb_agent( ID=10 )
    id_list = []
    while True:
        data = ser.readline()
        if data:
            id = get_uwb_id(data)
            range = get_uwb_range(data)
            print("Id: ", id, "  range: ", range)
            UAV_agent.handle_range_msg(Id=id, range=range)
            if not (id in id_list):
                id_list.append(id)
            if len(id_list) >= 3:
                print(UAV_agent.calc_pos_MSE())



#END
'''
