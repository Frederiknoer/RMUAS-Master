#!/usr/bin/env python3

import math
import numpy as np
import scipy as scipy
from numpy.random import uniform
import scipy.stats
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise, reshape_z
import sympy as symp


class KF:
    def __init__(self, xyz, v_ned, dt, option=0):
        self.option = option
        self.dt = dt

        if option == 0:
            r = 2.5 #1.4
            cu = 0.0001 #0.00002
        elif option == 1:
            r = 2.5
            cu = 0.025

        dim_x = 6
        dim_u = 3
        dim_z = 3
        #self.kf = KalmanFilter(dim_x=18, dim_z=3, dim_u=3)
        
        self.x = np.array([     xyz[0],
                                xyz[1],
                                xyz[2],
                                v_ned[0],
                                v_ned[1],
                                v_ned[2] ])
        
        
        self.R = np.eye(dim_z) # state uncertainty
        self.R *= r #7

        self.cov_u = np.eye(dim_u) # process uncertainty
        self.cov_u *= cu #0.0001 cov(u, u)
        #self.cov_u = Q_discrete_white_noise(dim=dim_u, dt=dt, var=0.12)
        #print(self.cov_u)

        self.P = np.eye(dim_x) # uncertainty covariance
        self.P *= 1000 # 500

        self.F = np.array([ [1, 0, 0, dt, 0, 0],
                            [0, 1, 0, 0, dt, 0],
                            [0, 0, 1, 0, 0, dt],
                            [0, 0, 0, 1, 0, 0 ],
                            [0, 0, 0, 0, 1, 0 ],
                            [0, 0, 0, 0, 0, 1 ] ])


        self.G = np.array([ [(dt**2)/2, 0,  0],
                            [0,  (dt**2)/2, 0],
                            [0,  0, (dt**2)/2],
                            [dt,  0,    0    ],
                            [0,  dt,    0    ],
                            [0,   0,   dt    ]  ])


        self.H = np.array([ [ 1, 0, 0, 0, 0, 0 ],
                            [ 0, 1, 0, 0, 0, 0 ],
                            [ 0, 0, 1, 0, 0, 0 ]  ])

        self.K = np.zeros((dim_x, dim_z)) # kalman gain

        self.I = np.eye(dim_x)
        
        #print("Kalman filter initialized, x_dim: ",self.x.shape, "  f_dim: ", self.F.shape, "  h_dim: ", self.H.shape, "  b_dim: ", self.G.shape)

    def get_return_vals(self):
        return self.R[0][0], self.cov_u[0][0]


    def predict(self, u):
        self.x = np.dot(self.F, self.x) + np.dot(self.G, u)

        FP = np.dot(self.F, self.P)
        FPFT = np.dot(FP, self.F.T)

        Gcov_u = np.dot(self.G, self.cov_u)
        Q = np.dot(Gcov_u, self.G.T)

        self.P = FPFT + Q
        

    def update(self, z):
        z = np.reshape(z, (3,))
        PHT = np.dot(self.P, self.H.T)
        S = np.linalg.inv( np.dot(self.H, PHT) + self.R )
        self.K = np.dot(PHT, S)

        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(self.K, y)

        KH = np.dot(self.K, self.H)
        self.P = np.dot((self.I - KH), self.P)


    def get_state(self):
        return self.x

    def get_plot_data(self):
        return np.sqrt( np.linalg.eig(self.P)[0][0:3] )
