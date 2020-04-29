#!/usr/bin/env python3


import math
import numpy as np
import scipy as scipy
from numpy.random import uniform
import scipy.stats
from filterpy.monte_carlo import resampling
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise, reshape_z
import sympy as symp
import serial
import localization as lx

class particleFilter:
    def __init__(self, start_vel, dt, anchors, option=0):
        # option 0 = standalone, option 1 = PKF
        self.option = option

        if option == 0: #PF
            self.N = 10000 #10000
            self.upd_std_dev = 2.2 #2.2
        else: #PKF:
            self.N = 7500 #10000
            self.upd_std_dev = 3.5 #2.2

        self.dt = dt
        self.anchors = anchors

        #Init particles and weights
        self.vel_x, self.vel_y, self.vel_z = start_vel

        self.weights = np.full((self.N, 1), 1.0)
        self.particles = np.zeros((self.N,3))
        self.particles[:,0] = np.random.uniform(-4.0, 4.0, size=self.N)
        self.particles[:,1] = np.random.uniform(-4.0, 4.0, size=self.N)
        self.particles[:,2] = np.random.uniform(-3.5, 0.5, size=self.N)

        #print("Particle Filter initiated with ", self.N, " Particles")

    def get_return_vals(self):
        return self.N, self.upd_std_dev

    def predict(self, u, v=None):
        if self.option == 0:
            mu, sigma = 0, 0.0005 #0.0002
            noise = np.random.normal(mu, sigma, 1)[0]
        else:
            mu, sigma = 0, 0.0005 #0.0002
            noise = np.random.normal(mu, sigma, 1)[0]
        self.particles[:, 0] += self.vel_x*self.dt + u[0]*((self.dt**2)/2) + noise
        self.particles[:, 1] += self.vel_y*self.dt + u[1]*((self.dt**2)/2) + noise
        self.particles[:, 2] += self.vel_z*self.dt + u[2]*((self.dt**2)/2) + noise
        if v == None:
            self.vel_x += u[0]*self.dt
            self.vel_y += u[1]*self.dt
            self.vel_z += u[2]*self.dt
        else:
            self.vel_x = v[0]
            self.vel_y = v[1]
            self.vel_z = v[2]

    def update(self, z, anchs=0):
        if anchs != 0:
            self.anchors = anchs
        p = self.particles
        for i, anchor in enumerate(self.anchors):
            est_dist = (((p[:,0] - anchor[0])**2 + (p[:,1]-anchor[1])**2 + (p[:,2]-anchor[2])**2)**0.5)
            prob = scipy.stats.norm( est_dist, self.upd_std_dev ).pdf(z[i])
            prob = np.resize(prob,(self.N, 1))
            self.weights = np.multiply(self.weights, prob) 

        self.weights = np.true_divide(self.weights, np.sum(self.weights))

    def resample(self):
        indexes = resampling.systematic_resample(self.weights)
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights = np.true_divide(self.weights, np.sum(self.weights))

    def estimate(self):
        #return np.mean(self.particles, axis=0)
        #max_idx = np.argmax(self.weights)
        idx = np.argsort(self.weights)[250:]
        return np.mean(self.particles, axis=0)
        #return self.particles[max_idx]

    def get_particles(self):
        return self.particles

'''
if __name__ == "__main__":
    d = 4
    dy = d * (np.sqrt(3)/2)

    A = np.array([0.0, 0.0, 0.0])
    B = np.array([d  , 0.0, 0.0])
    C = np.array([d/2, dy, 0.0])
    #D = np.array([d/2, -dy, 0.1])
    #E = np.array([-(d/2), dy, 0.0])
    #F = np.array([-(d), 0.0, 0.0])
    #G = np.array([-(d/2), -dy, 0.05])

    anchs = [A,B,C] #,D,E,F,G]

    pf = PF(dt=0.02, anchors=anchs)
    #pf.predict()
    pf.update([1.2, 0.3, 0.9])
    print(pf.estimate())
'''