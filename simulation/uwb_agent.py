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
from filterpy.common import Q_discrete_white_noise, reshape_z
import sympy as symp
import scipy as scip
import serial


PI = 3.14159265359
DEBUG = False

class KF:
    def __init__(self):
        self.dt = 5e-2
        self.prev_time = 0

        #self.kf = KalmanFilter(dim_x=18, dim_z=3, dim_u=3)
        self.x = np.array([     2.00,
                                1.50,
                                0.00,
                                0.10,
                                0.10,
                                0.10,

                                -1.0,
                                1.50,
                                0.00,
                                0.10,
                                0.10,
                                0.10,

                                0.5,
                                -0.25,
                                0.0,
                                0.10,
                                0.10,
                                0.10 ])
        
        self.R = np.eye(3)
        self.R *= 0.05

        self.Q = np.eye(3)
        self.Q *= 0.8

        self.P = np.eye(18)
        self.P *= 500

        self.calc_H()
        
        self.calc_F_G(self.dt)

        self.K = np.array([0,0,0])

        print("Kalman filter initialized, x_dim: ",self.x.shape, "  f_dim: ", self.F.shape, "  h_dim: ", self.H.shape, "  b_dim: ", self.G.shape)

    def calc_H(self):
        p_mag1 = np.linalg.norm(self.x[0:3])
        p_mag2 = np.linalg.norm(self.x[6:9])
        p_mag3 = np.linalg.norm(self.x[12:15])
        self.H = np.array([     [ (self.x[0] / p_mag1), (self.x[1] / p_mag1), (self.x[2] / p_mag1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                                [ 0, 0, 0, 0, 0, 0, (self.x[6] / p_mag2), (self.x[7] / p_mag2), (self.x[8] / p_mag2), 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (self.x[12] / p_mag3), (self.x[13] / p_mag3), (self.x[14] / p_mag3), 0, 0, 0 ] ])


    def calc_F_G(self, dt):
        self.F = np.array([     [1.0, 0.0, 0.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] ])


        self.G = np.array([     [(dt**2)/2, 0.0, 0.0],
                                [0.0, (dt**2)/2, 0.0],
                                [0.0, 0.0, (dt**2)/2],
                                [dt,  0.0 ,    0.0  ],
                                [0.0,   dt,    0.0  ],
                                [0.0,    0.0,   dt  ],

                                [(dt**2)/2, 0.0, 0.0],
                                [0.0, (dt**2)/2, 0.0],
                                [0.0, 0.0, (dt**2)/2],
                                [dt,  0.0 ,    0.0  ],
                                [0.0,   dt,    0.0  ],
                                [0.0,    0.0,   dt  ],

                                [(dt**2)/2, 0.0, 0.0],
                                [0.0, (dt**2)/2, 0.0],
                                [0.0, 0.0, (dt**2)/2],
                                [dt,  0.0 ,    0.0  ],
                                [0.0,   dt,    0.0  ],
                                [0.0,    0.0,   dt  ]  ])



    def predict(self, acc_in, t_in):
        #dt = t_in - self.prev_time
        #self.prev_time = t_in
        #self.calc_F_G(dt)

        self.x = np.dot(self.F, self.x) + np.dot(self.G, acc_in)

        FP = np.dot(self.F, self.P)
        FPFT = np.dot(FP, self.F.T)

        GQ = np.dot(self.G, self.Q)
        GQGT = np.dot(GQ, self.G.T)

        self.P = FPFT + GQGT 
        

    def update(self, z):
        self.calc_H()

        PHT = np.dot(self.P, self.H.T)
        S = np.linalg.inv( np.dot(self.H, PHT) + self.R )
        self.K = np.dot(PHT, S)

        zhx = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(self.K, zhx)

        KH = np.dot(self.K, self.H)

        self.P = self.P - (np.dot( KH, self.P ))


    def get_state(self):
        return self.x


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

        self.avg_range_arr = np.empty([ 4,5 ])

        #self.ID_list = np.array(["FFFFE3BC", "FFFFA858", "6F0B"])
        if self.id == 10:
            self.UAV_KF = KF()
            self.kf_range_in = np.array([0,0,0])
            self.range_check = np.array([False,False,False])

    def mvg_avg(self, id, range):
        N = 5
        self.avg_range_arr[id] = np.roll(self.avg_range_arr[id], 1)
        self.avg_range_arr[id][0] = range
        #print(self.avg_range_arr[id])
        return np.average(self.avg_range_arr[id])
        


    def handle_range_msg(self, Id, range):
        range_f = self.mvg_avg(Id, range)
        #print("range, range filtered")
        #print(range, range_f)
        self.add_nb_module(Id, range_f)
        if Id != 3:
            self.kf_range_in[Id] = range
            self.range_check[Id] = True
            if not any(x == False for x in self.range_check):
                print self.range_check
                self.UAV_KF.update(self.kf_range_in)
                self.range_check[:] = False
        #self.update_incidenceMatrix()

    def handle_other_msg(self, Id1, Id2, range):
        self.add_pair(Id1, Id2, range)
        #self.update_incidenceMatrix()

    def handle_acc_msg(self, acc_in, t_in):
        self.UAV_KF.predict(acc_in, t_in)

    def get_kf_state(self):
        return self.UAV_KF.get_state()

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
            mse += (distance_calculated - distance)**2
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

        initial_location = np.array([2.0, 1.5, 0.5])

        result = scip.optimize.minimize(
            self.mse,                    # The error function
            initial_location,            # The initial guess
            args=(locations, distances), # Additional parameters for mse
            method='L-BFGS-B',           # The optimisation algorithm
            options={
            'ftol':1e-6,                 # Tolerance
            'maxiter': 1e+7              # Maximum iterations
            })
        location = result.x

        return location

    def predefine_ground_plane(self):
        A = np.array([0.0, 0.0, 0.0])
        B = np.array([3.0, 0.0, 0.0])
        C = np.array([1.5, 1.75, 0.0])
        D = np.array([1.5, -1.75, 0.0])

        self.poslist = np.array([A,B,C,D])
        return A, B, C, D


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
            range = 0.9 * get_uwb_range(data) - 23.52
            print("Id: ", id, "  range: ", range)
            UAV_agent.handle_range_msg(Id=id, range=range)

            if not (id in id_list):
                id_list.append(id)
            if len(id_list) >= 3:
                print(UAV_agent.calc_pos_MSE())



#END
'''
