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
    def __init__(self, start_vel, dt, anchors):
        self.N = 7500
        self.dt = dt
        self.anchors = anchors

        #Init particles and weights
        self.vel_x, self.vel_y, self.vel_z = start_vel

        self.particles = np.empty(( self.N, 3 ))
        self.weights = np.full((self.N, 1), 1.0)
        self.particles = np.random.uniform(-8.0,8.0,(self.N,3))

        print("Particle Filter initiated with ", self.N, " Particles")

    def predict(self, u):
        #mu, sigma = 0, 0.005
        self.particles[:, 0] += self.vel_x*self.dt + u[0]*((self.dt**2)/2) #+ np.random.normal(mu, sigma, 1)[0]
        self.particles[:, 1] += self.vel_y*self.dt + u[1]*((self.dt**2)/2) #+ np.random.normal(mu, sigma, 1)[0]
        self.particles[:, 2] += self.vel_z*self.dt + u[2]*((self.dt**2)/2) #+ np.random.normal(mu, sigma, 1)[0]
        self.vel_x += u[0]*self.dt
        self.vel_y += u[1]*self.dt
        self.vel_z += u[2]*self.dt

    def update(self, z):
        #mu, sigma = 0, 0.02
        p = self.particles
        for i, anchor in enumerate(self.anchors):
            est_dist = (((p[:,0] - anchor[0])**2 + (p[:,1]-anchor[1])**2 + (p[:,2]-anchor[2])**2)**0.5) #+ np.random.normal(mu, sigma, 1)[0]
            prob = scipy.stats.norm(est_dist, 1.5).pdf(z[i])
            prob = np.resize(prob,(self.N, 1))
            self.weights = np.multiply(self.weights, prob)

        self.weights = np.true_divide(self.weights, np.sum(self.weights))

    def resample(self):
        indexes = resampling.systematic_resample(self.weights)
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights = np.true_divide(self.weights, np.sum(self.weights))

    def estimate(self):
        return np.mean(self.particles, axis=0)

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