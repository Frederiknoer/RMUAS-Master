#!/usr/bin/env python3

import math
import numpy as np
import scipy as scipy
from numpy.random import uniform
import scipy.stats
import sympy as symp
import serial
import localization as lx
import particleFilter as PF
import kalmanFilter as KF
import time

'''
Position Rules:
ID = 0: Origin of the local coordinate system (a)
ID = 1: Always 0 in y- and z-components (b)
ID = 2: Always posetive in both x- and y-components
ID = 3: Always posetive in x-component and negative in y-component
ID = 10: Always the UAV
'''

class uwb_agent:
    def __init__(self, ID, d=None):
        self.id = int(ID)
        self.d = d

        self.M = np.array([self.id]) #Modules
        self.N = np.array([]) #Neigbours
        self.E = np.array([0]) #
        self.pairs = np.empty((0,3))
        self.poslist = np.array([])

        #MSE:
        self.prev_val = np.array([1.5, 2.0, 0.1])

        #FILTERS:
        self.KF_started = False

        #TIME HANDLERS:
        self.time_taken_pf_upd = 0.0
        self.time_instanes_pf_upd = 0
        self.time_taken_pf_predict = 0.0
        self.time_instanes_pf_predict = 0
        self.time_taken_pf_resamp = 0.0
        self.time_instanes_pf_resamp = 0

        self.time_taken_kf_upd = 0.0
        self.time_instanes_kf_upd = 0
        self.time_taken_kf_predict = 0.0
        self.time_instanes_kf_predict = 0

        self.time_taken_geo = 0.0
        self.time_instanes_geo = 0

    # ***************** RETURN TIME VALUES ************************
    def get_time_vals(self, method):
        if method == 'NF' or method == 'NF4':
            return ((self.time_taken_geo) / (self.time_instanes_geo))
        elif method == 'KF' or method == 'KF4':
            return [((self.time_taken_kf_predict) / (self.time_instanes_kf_predict)) ,  ((self.time_taken_kf_upd) / (self.time_instanes_kf_upd))]
        elif method == 'PF' or method == 'PF4':
            return [((self.time_taken_pf_predict) / (self.time_instanes_pf_predict)) ,  ((self.time_taken_pf_upd) / (self.time_instanes_pf_upd)) , ((self.time_taken_pf_resamp) / (self.time_instanes_pf_resamp))]
        elif method == 'PKF' or method == 'PKF4':
            pass

    # ***************** PARTICLE FILTER FUNCTIONS *****************
    def startPF(self, start_vel, dt, option=0):
        self.anchors = self.predefine_ground_plane()
        self.PF = PF.particleFilter(dt=dt, start_vel=start_vel, anchors=self.anchors, option=option)
        return self.PF.get_return_vals()

    def PFpredict(self, u, v=None):
        prev_t = time.time()
        self.PF.predict(u)
        self.time_taken_pf_predict += time.time() - prev_t
        self.time_instanes_pf_predict += 1

    def PFupdate(self, use4):
        prev_t = time.time()
        z = self.get_ranges()
        n=0
        if use4:
            a,b,c,d,e,f,g = self.anchors
            n = [a,b,c,d,e,f,g]
            n,z = self.get_x_closest_nodes(n, z, x=4)

        self.PF.update(z=z, anchs=n, use4=use4)
        self.time_taken_pf_upd += time.time() - prev_t
        self.time_instanes_pf_upd += 1

        prev_t = time.time()
        self.PF.resample()
        self.time_taken_pf_resamp += time.time() - prev_t
        self.time_instanes_pf_resamp += 1
        
    def getPFpos(self):
        return self.PF.estimate()

    def get_particles(self):
        return self.PF.get_particles()

    # ***************** KALMAN FILTER FUNCTIONS *****************
    def startKF(self, xyz, v_ned, dt, option=0):
        self.UAV_KF = KF.KF(xyz, v_ned, dt, option=option)
        self.KF_started = True
        return self.UAV_KF.get_return_vals()

    def KFpredict(self, acc):
        prev_t = time.time()
        self.UAV_KF.predict(acc)
        self.time_taken_kf_predict += time.time() - prev_t
        self.time_instanes_kf_predict += 1

    def get_kf_state(self):
        return self.UAV_KF.get_state()[0:3]


    # ***************** KALMAN PARTICLE FILTER FUNCTIONS *****************
    def startPKF(self, acc, dt, xyz, v_ned):
        #print("STARTING PARTICLE KALMAN FILTER")
        pf_val1, pf_val2 = self.startPF(v_ned, dt, option=1)
        kf_val1, kf_val2 = self.startKF(xyz, v_ned, dt, option=1)
        return kf_val1, kf_val2, pf_val1, pf_val2

    def predictPKF(self, u):
        self.UAV_KF.predict(u)
        kf_v = self.get_kf_state()[3:6]
        self.PFpredict(u=u, v=kf_v)

    def updatePKF(self, use4):
        self.PFupdate(use4)
        pf_pos = self.getPFpos()
        self.UAV_KF.update(z=pf_pos)

    def get_PKFstate(self):
        #print(self.UAV_KF.get_state()[0:3])
        return self.UAV_KF.get_state()[0:3]


    # ***************** HANDLE INPUT FUNCTIONS *****************
    def handle_range_msg(self, Id, range):
        self.add_nb_module(Id, range)

    def handle_other_msg(self, Id1, Id2, range):
        self.add_pair(Id1, Id2, range)

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

    # ***************** UTILITY FUNCTIONS *****************
    def calc_dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def clean_cos(self, cos_angle):
        return min(1,max(cos_angle,-1))

    def get_ranges(self):
        r = np.empty(7)
        for i, pair in enumerate(self.pairs):
            if pair[0] == 10:
                if pair[1] == 0:
                    r[0] = pair[2]
                elif pair[1] == 1:
                    r[1] = pair[2]
                elif pair[1] == 2:
                    r[2] = pair[2]
                elif pair[1] == 3:
                    r[3] = pair[2]
                elif pair[1] == 4:
                    r[4] = pair[2]
                elif pair[1] == 5:
                    r[5] = pair[2]
                elif pair[1] == 6:
                    r[6] = pair[2]
        return r

    def predefine_ground_plane(self):
        d = self.d
        dy = d * (np.sqrt(3)/2)

        A = np.array([0.0, 0.0, 0.0])
        B = np.array([d  , 0.0, 0.0])
        C = np.array([d/2, dy, 0.0])
        D = np.array([d/2, -dy, 0.1])
        E = np.array([-(d/2), dy, 0.0])
        F = np.array([-(d), 0.0, 0.0])
        G = np.array([-(d/2), -dy, 0.05])

        self.poslist = [A,B,C,D,E,F,G]
        return A, B, C, D, E, F, G

    def get_x_closest_nodes(self, n, r, x=4):
        idx = np.argsort(r)
        new_r = np.empty(len(r))
        new_n = np.empty((len(n),3))
        #print("array before sort(r): \n", r, "\n(n): \n", n)
        for i in range(len(r-1)):
            new_r[i] = r[idx[i]]
            new_n[i] = n[idx[i]]
        #print("array after sort(r): \n", new_r[0:x], "\n(n): \n", new_n[0:x])
        return new_n[0:x], new_r[0:x]

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


    # ***************** POSITION ESTIMATION FUNCTIONS *****************
    def calc_pos_alg(self, use4):
        prev_t = time.time()
        #use4 = False
        if use4:
            nodes = 4
        else:
            nodes = 7
        a,b,c,d,e,f,g = self.predefine_ground_plane() #A, B, C, D, E, F, G
        x = np.array([ a[0], b[0], c[0], d[0], e[0], f[0], g[0] ])
        y = np.array([ a[1], b[1], c[1], d[1], e[1], f[1], g[1] ])
        z = np.array([ a[2], b[2], c[2], d[2], e[2], f[2], g[2] ])

        A = np.zeros((nodes-1, 3))
        B = np.zeros(nodes-1)

        X = np.zeros(3)

        q = np.zeros(nodes)
        r = self.get_ranges()
        n = np.array([a,b,c,d,e,f,g])

        #if use4:
            #n, r = self.get_x_closest_nodes(n,r)

        for i in range(nodes):
            q[i] = (r[i]**2 - x[i]**2 - y[i]**2 - z[i]**2) / 2

        
        for i in range(nodes-1):
            #print i 
            A[i][0] = x[0] - x[i+1]
            A[i][1] = y[0] - y[i+1]
            A[i][2] = z[0] - z[i+1]

            B[i] = q[i+1] - q[0]

        if use4:
            X = np.dot(np.linalg.inv(A), B)
        else:
            X = np.linalg.lstsq(A, B, rcond=None)[0]
        
        self.time_taken_geo += time.time() - prev_t
        self.time_instanes_geo += 1

        if self.KF_started:
            prev_t = time.time()
            self.UAV_KF.update(z=X)
            self.time_taken_kf_upd += time.time() - prev_t
            self.time_instanes_kf_upd += 1
            return self.UAV_KF.get_state()[0:3]
        else:
            return X


    def mse(self, x, c, r):
        mse = 0.0
        for location, distance in zip(c, r):
            distance_calculated = self.calc_dist(x,location)
            mse += (distance_calculated - distance)**2
        return mse / len(c)

    def calc_pos_MSE(self):
        c = self.predefine_ground_plane()
        c = c[0:3]
        r = self.get_ranges()
        r = r[0:3]

        x0 = self.prev_val

        res = scipy.optimize.minimize(self.mse, x0, args=(c, r), method='SLSQP')
        self.prev_val = res.x

        return [self.prev_val[0], self.prev_val[1], -(abs(self.prev_val[2]))]

    def get_uwb_id(self, data):
        id_idx = data.find('from:')
        if id_idx == -1:
            return None
        id = ''
        while data[id_idx+6] != ' ':
            id += data[id_idx+6]
            id_idx += 1
        id = id[:-1]
        return id

    def get_uwb_range(self, data):
        range_idx = data.find('Range:')
        if range_idx == -1:
            return None
        data_val = (float(data[range_idx+7])) + (float(data[range_idx+9])*0.1) + (float(data[range_idx+10])*0.01)
        return data_val


if __name__ == "__main__":
    agent = uwb_agent(10)
    ser = serial.Serial('/dev/ttyUSB0', 115200)

    n_of_anchors = 1
    range_list = np.array([])
    id_list    = np.array([])
    print()
    
    while True:
        data = ser.readline().decode("utf-8")
        print("Raw Data: ", data)
        if data[0] == 'f':
            id = int(agent.get_uwb_id(data),16)
            range = agent.get_uwb_range(data)

            if id_list.size == 0:
                id_list = np.append(id_list, id)
                range_list = np.append(range_list, range)
            elif not any(x == id for x in id_list):
                id_list = np.append(id_list, id)
                range_list = np.append(range_list, range)
            else:
                for i, elem in enumerate(id_list):
                    if elem == id:
                        range_list[i] = range


            print("id list: ", id_list)
            print("range list: ", range_list)

            
            
                

            




