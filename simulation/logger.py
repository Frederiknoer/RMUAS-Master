#!/usr/bin/env python3

from scipy import linalg as la
import matplotlib.pyplot as pl
import numpy as np
import math
import sys
import random

import pycopter as pycopter_class

sys.path.append("pycopter/")
import quadlog
import animation as ani


n_of_sims = 2


class logger:
    def __init__(self):
        self.N = n_of_sims
        self.tf = 50
        self.dt = 1/100
        self.time = np.linspace(0, self.tf, int(self.tf/self.dt))

        self.run_animation = False
        #print(sys.argv[1])
        self.pos_method = sys.argv[1] #NF4 = No Filter(4 closest), NF = No filter(svd), KF, KF4 = kalman filter, PF = particle filter, PKF = particle kalman filter

        #LOGS:
        self.big_log_Ed  = np.empty([int(self.tf/self.dt), 1, 0], dtype=np.float16)
        self.big_log_est = np.empty([3, int(self.tf/self.dt), 0], dtype=np.float16)
        self.big_log_gt  = np.empty([3, int(self.tf/self.dt), 0], dtype=np.float16)

        self.pycopter = pycopter_class.pycopter(self.tf, self.dt)


    def run_logger(self):
        for i in range(self.N):
            UAV, alg, ed = self.pycopter.run(method=self.pos_method, run_animation=self.run_animation) 
            
            self.big_log_Ed = np.insert(arr=self.big_log_Ed, obj=i, values=ed, axis=2)

            self.big_log_est = np.insert(arr=self.big_log_est, obj=i, values=alg, axis=2)
            self.big_log_gt = np.insert(arr=self.big_log_gt, obj=i, values=UAV, axis=2)
            #del self.pycopter

    
    def calc_statistics(self):
        #Calculate statistics:
        print(self.big_log_Ed)
        self.Ed_mean  = np.mean(self.big_log_Ed, axis=2)
        self.Ed_var   = np.var(self.big_log_Ed, axis=2)
        print("mean shape:", self.Ed_mean.shape)

        self.est_mean = np.mean(self.big_log_est, axis=2)
        self.est_var  = np.var(self.big_log_est, axis=2)

        self.gt_mean  = np.mean(self.big_log_gt, axis=2)
        self.gt_var   = np.var(self.big_log_gt, axis=2)

    '''
    def parse_data(self):
        self.Ed_mean  = self.big_log_Ed[0][:]
        self.alt_mean = self.big_log_alt[0][:]
        self.x_mean   = self.big_log_x[0][:]
        self.y_mean   = self.big_log_y[0][:]
    '''

    def plot(self):
        n_of_particles, std_add, Q, R =  self.pycopter.n_of_particles, self.pycopter.std_add, self.pycopter.Q, self.pycopter.R

        method = self.pos_method
        quadcolor = ['r', 'g', 'b']
        pl.close("all")
        pl.ion()

        print(self.time.shape)
        print(self.Ed_mean.shape)
        
        if method == 'NF':
            info1 = info2 = ''
        elif method == 'KF':
            info1 = 'R: ' + str(R)
            info2 = 'Q: ' + str(Q)
        elif method == 'PF':
            info1 = 'Particles: ' + str(n_of_particles)
            info2 = 'Sigma P: ' + str(std_add)
        elif method == 'PKF':
            info1 = 'R: ' + str(R)
            info2 = 'Q: ' + str(Q)
            info3 = 'Particles: ' + str(n_of_particles)
            info4 = 'Sigma P: ' + str(std_add)

        pl.figure(1)
        if method == 'PKF':
            pl.title(method +" 2D Pos[m] - " + info1 + " - " + info2 + "\n" + info3 + " - " + info4)
        else:
            pl.title(method +" 2D Pos[m] - " + info1 + " - " + info2)
        pl.plot(self.est_mean[0,:], self.est_mean[1,:], label="est_pos(x,y)", color=quadcolor[2])
        pl.plot(self.gt_mean[0,:], self.gt_mean[1, :], label="Ground Truth(x,y)", color=quadcolor[0])
        pl.xlabel("East")
        pl.ylabel("South")
        pl.legend()
        pl.savefig('results/'+method+'_2D_pos.png')
        
        pl.figure(2)
        if method == 'PKF':
            pl.title(method+" Error Dist[m] - " + info1 + " - " + info2 + "\n" + info3 + " - " + info4)
        else:
            pl.title(method+" Error Dist[m] - " + info1 + " - " + info2)
        pl.plot(self.time, self.Ed_mean, label="Distance: est_pos - true_pos", color=quadcolor[2])
        #pl.ylim(-0.1,1)
        pl.xlabel("Time [s]")
        pl.ylabel("Formation distance error [m]")
        pl.grid()
        pl.legend()
        pl.savefig('results/'+method+'_err_pos.png')
        
        

        pl.figure(3)
        if method == 'PKF':
            pl.title(method+" Altitude[m] - " + info1 + " - " + info2 + "\n" + info3 + " - " + info4)
        else:
            pl.title(method+" Altitude[m] - " + info1 + " - " + info2)
        pl.plot(self.time, self.est_mean[2,:], label="est_alt", color=quadcolor[2])
        pl.plot(self.time, self.gt_mean[2,:], label="Ground Truth(alt)", color=quadcolor[0])
        pl.ylim(-4, 0.5)
        pl.xlabel("Time [s]")
        pl.ylabel("Altitude [m]")
        pl.grid()
        pl.legend(loc=2)
        pl.savefig('results/'+method+'_alt.png')

        #pl.pause(0)


if __name__ == "__main__":
    l = logger()
    l.run_logger()
    l.calc_statistics()
    l.plot()
