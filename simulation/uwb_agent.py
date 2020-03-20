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
import localization as lx


PI = 3.14159265359
DEBUG = False

class KF:
    def __init__(self, r_in, v_ned, acc):
        self.dt = 0.2
        n_of_nodes = 3

        dim_x = n_of_nodes + 3
        dim_u = 3
        dim_z = n_of_nodes
        #self.kf = KalmanFilter(dim_x=18, dim_z=3, dim_u=3)
        
        self.x = np.array([     r_in[0],
                                r_in[1],
                                r_in[2],
                                v_ned[0],
                                v_ned[1],
                                v_ned[2] ])
        
        
        self.R = np.eye(dim_z) # state uncertainty
        self.R *= 7 #7

        self.cov_u = np.eye(dim_u) # process uncertainty
        self.cov_u *= 0.02 #0.0001 cov(u, u)

        self.P = np.eye(dim_x) # uncertainty covariance
        self.P *= 1000 # 500
        
        self.calc_F_G(self.dt, a=acc)
        self.calc_H()

        self.K = np.zeros((dim_x, dim_z)) # kalman gain

        self.I = np.eye(dim_x)
        
        print("Kalman filter initialized, x_dim: ",self.x.shape, "  f_dim: ", self.F.shape, "  h_dim: ", self.H.shape, "  b_dim: ", self.G.shape)

    def calc_H(self):
        self.H = np.array([ [ 1, 0, 0, 0, 0, 0 ],
                            [ 0, 1, 0, 0, 0, 0 ],
                            [ 0, 0, 1, 0, 0, 0 ]  ])


    def calc_F_G(self, dt, a):
        v_mag = np.linalg.norm( self.x[3:6] )
        a_mag = np.linalg.norm(a)
        vx, vy, vz = self.x[3:6]
        ax, ay, az = a

        self.F = np.array([ [1, 0, 0, vx*(dt/v_mag), vy*(dt/v_mag),  vz*(dt/v_mag)],
                            [0, 1, 0, vx*(dt/v_mag), vy*(dt/v_mag),  vz*(dt/v_mag)],
                            [0, 0, 1, vx*(dt/v_mag), vy*(dt/v_mag),  vz*(dt/v_mag)],
                            [0, 0, 0,      1,             0,               0      ],
                            [0, 0, 0,      0,             1,               0      ],
                            [0, 0, 0,      0,             0,               1      ] ])


        self.G = np.array([ [ax*(((dt**2)/2)/a_mag), ay*(((dt**2)/2)/a_mag), az*(((dt**2)/2)/a_mag)],
                            [ax*(((dt**2)/2)/a_mag), ay*(((dt**2)/2)/a_mag), az*(((dt**2)/2)/a_mag)],
                            [ax*(((dt**2)/2)/a_mag), ay*(((dt**2)/2)/a_mag), az*(((dt**2)/2)/a_mag)],
                            [dt,                               0,                      0           ],
                            [0,                               dt,                      0           ],
                            [0,                                0,                     dt           ]  ])



    def predict(self, u):
        print("Acc:", u)
        self.calc_F_G(self.dt, u)
        
        self.x = np.dot(self.F, self.x) + np.dot(self.G, u)

        FP = np.dot(self.F, self.P)
        FPFT = np.dot(FP, self.F.T)

        Gcov_u = np.dot(self.G, self.cov_u)
        Q = np.dot(Gcov_u, self.G.T)

        self.P = FPFT + Q
        #print("Old Eigen P: ", np.linalg.eig(self.P)[0][0:3] )
        

    def update(self, z):
        PHT = np.dot(self.P, self.H.T)
        S = np.linalg.inv( np.dot(self.H, PHT) + self.R )
        self.K = np.dot(PHT, S)

        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(self.K, y)

        KH = np.dot(self.K, self.H)
        self.P = np.dot((self.I - KH), self.P)
        
        #self.P = self.P - (np.dot( KH, self.P ))
        #print("New Eigin P: ", np.linalg.eig(self.P)[0][0:3] )

    def get_state(self, id):
        if id == 100:
            return self.x[3:6]
        else:
            return self.x[id]

    def get_plot_data(self):
        return np.sqrt( np.linalg.eig(self.P)[0][0:3] )


class uwb_agent:
    def __init__(self, ID, d=None):
        self.id = int(ID)
        self.d = d

        self.M = np.array([self.id]) #Modules
        self.N = np.array([]) #Neigbours
        self.E = np.array([0]) #
        self.P = np.array([]) #
        self.pairs = np.empty((0,3))

        self.val = np.array([])
        self.prev_val = np.array([1.5, 2.0, 0.1])

        self.poslist = np.array([])

        self.avg_range_arr = np.empty([ 7,4 ])

        self.KF_started = False

    def startKF(self, v_ned, acc):
        r = self.get_ranges()
        self.kf_range_in = np.array([r[0], r[1], r[2]])

        self.UAV_KF = KF(self.kf_range_in, v_ned, acc)
        
        self.range_check = np.array([False,False,False])
        self.KF_started = True

    
    def get_plot_data(self):
        return self.UAV_KF.get_plot_data()


    def mvg_avg(self, id, range):
        self.avg_range_arr[id] = np.roll(self.avg_range_arr[id], 1)
        self.avg_range_arr[id][0] = range
        return np.average(self.avg_range_arr[id])
        


    def handle_range_msg(self, Id, range):
        #range_f = self.mvg_avg(Id, range)
        #print("range, range filtered")
        #print(range, range_f)
        
        if self.KF_started:
            #self.UAV_KF.update(range)
            kf_r = self.UAV_KF.get_state(Id)
            self.add_nb_module(Id, kf_r)
        
            self.kf_range_in[Id] = range
            self.range_check[Id] = True
            if not any(x == False for x in self.range_check):
                #self.UAV_KF.update(self.kf_range_in)
                self.range_check[:] = False
        else:
            self.add_nb_module(Id, range)
        

    def handle_other_msg(self, Id1, Id2, range):
        self.add_pair(Id1, Id2, range)

    def handle_acc_msg(self, acc_in):
        self.UAV_KF.predict(acc_in)

    def get_kf_state(self, Id):
        return self.UAV_KF.get_state(Id)

    def calc_dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)


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

    def get_ranges(self):
        r = [None]*7
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


    def calc_pos_LS(self):
        nodes = 7
        a,b,c,d,e,f,g = self.predefine_ground_plane() #A, B, C, D, E, F, G
        x = np.array([ a[0], b[0], c[0], d[0], e[0], f[0], g[0] ])
        y = np.array([ a[1], b[1], c[1], d[1], e[1], f[1], g[1] ])
        z = np.array([ a[2], b[2], c[2], d[2], e[2], f[2], g[2] ])

        A = np.zeros((nodes, 3))
        B = np.zeros(nodes)
        r = self.get_ranges()

        n = nodes-2
        for i in range(nodes):
            A[i][0] = 2*x[n] - 2*x[i]
            A[i][1] = 2*y[n] - 2*y[i]
            A[i][2] = 2*z[n] - 2*z[i]

            B[i] = r[i]**2 - r[n]**2 - x[i]**2 - y[i]**2 - z[i]**2 + x[n]**2 + y[n]**2 + z[n]**2

        '''
        print('A:')
        print(A)
        print('B:')
        print(B)
        '''

        res = np.linalg.lstsq(A,B)
        #print(res)
        return res[0]
        
        


    def clean_cos(self, cos_angle):
        return min(1,max(cos_angle,-1))

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

        res = scip.optimize.minimize(self.mse, x0, args=(c, r), method='SLSQP')
        self.prev_val = res.x

        return [self.prev_val[0], self.prev_val[1], -(abs(self.prev_val[2]))]

    def calc_pos_TRI(self):
        pass

    def predefine_ground_plane(self):
        d = self.d
        dy = d * (np.sqrt(3)/2)

        A = np.array([0.0, 0.0, 0.0])
        B = np.array([d  , 0.0, 0.0])
        C = np.array([d/2, dy, 0.0])
        D = np.array([d/2, -dy, 0.0])
        E = np.array([-(d/2), dy, 0.0])
        F = np.array([-(d), 0.0, 0.0])
        G = np.array([-(d/2), -dy, 0.0])

        self.poslist = [A,B,C,D,E,F,G]
        return A, B, C, D, E, F, G


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
