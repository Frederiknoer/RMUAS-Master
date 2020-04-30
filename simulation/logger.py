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

import warnings
warnings.simplefilter("ignore")


n_of_sims = 25


class logger:
    def __init__(self, method):
        print("************************* METHOD: ", method, " **************************")
        self.N = n_of_sims
        self.tf = 350
        self.dt = 1/100
        self.time = np.linspace(0, self.tf, int(self.tf/self.dt))

        self.run_animation = False
        #print(sys.argv[1])
        self.pos_method = method #sys.argv[1] #NF4 = No Filter(4 closest), NF = No filter(svd), KF, KF4 = kalman filter, PF = particle filter, PKF = particle kalman filter

        self.n = int(self.tf/self.dt)

        #LOGS:
        self.big_log_Ed  = np.empty([self.n, 1, 0], dtype=np.float32)
        self.big_log_est = np.empty([3, self.n, 0], dtype=np.float32)
        self.big_log_gt  = np.empty([3, self.n, 0], dtype=np.float32)

        self.big_log_time = np.empty([1,0])

        #self.pycopter = pycopter_class.pycopter(self.tf, self.dt)


    def run_logger(self):
        for i in range(self.N):
            print("Starting simulation #", i+1, " of #", self.N)
            self.pycopter = pycopter_class.pycopter(self.tf, self.dt)
            UAV, alg, ed = self.pycopter.run(method=self.pos_method, run_animation=self.run_animation) 
            
            self.big_log_Ed = np.insert(arr=self.big_log_Ed, obj=i, values=ed, axis=2)
            self.big_log_est = np.insert(arr=self.big_log_est, obj=i, values=alg, axis=2)
            self.big_log_gt = np.insert(arr=self.big_log_gt, obj=i, values=UAV, axis=2)

            self.big_log_time = np.insert(arr=self.big_log_time, values=(self.pycopter.UAV_agent.get_time_vals(self.pos_method)), obj=i, axis=1)

            self.n_of_particles, self.std_add, self.Q, self.R =  self.pycopter.n_of_particles, self.pycopter.std_add, self.pycopter.Q, self.pycopter.R
            del self.pycopter

    
    def calc_statistics(self):
        #Calculate statistics:
        self.Ed_mean  = np.mean(self.big_log_Ed, axis=2)
        self.Ed_var   = np.var(self.big_log_Ed, axis=2)

        self.est_mean = np.mean(self.big_log_est, axis=2)
        self.est_var  = np.var(self.big_log_est, axis=2)

        self.gt_mean  = np.mean(self.big_log_gt, axis=2)
        self.gt_var   = np.var(self.big_log_gt, axis=2)

        self.time_mean = np.mean(self.big_log_time, axis=1)
        self.time_var = np.var(self.big_log_time, axis=1)
        print("Mean of Operation Time: ", self.time_mean)
        print("Var of Operation Time: ", self.time_var)

    '''
    def parse_data(self):
        self.Ed_mean  = self.big_log_Ed[0][:]
        self.alt_mean = self.big_log_alt[0][:]
        self.x_mean   = self.big_log_x[0][:]
        self.y_mean   = self.big_log_y[0][:]
    '''

    def plot(self):
        n_of_particles, std_add, Q, R =  self.n_of_particles, self.std_add, self.Q, self.R

        method = self.pos_method
        quadcolor = ['r', 'g', 'b']
        pl.close("all")
        pl.ion()

        #print(self.time.shape)
        #print(self.Ed_mean.shape)
        
        if method == 'NF' or method == 'NF4':
            info1 = info2 = ''
        elif method == 'KF' or method == 'KF4':
            info1 = 'R: ' + str(R)
            info2 = 'Q: ' + str(Q)
        elif method == 'PF' or method == 'PF4':
            info1 = 'Particles: ' + str(n_of_particles)
            info2 = 'Sigma P: ' + str(std_add)
        elif method == 'PKF' or method == 'PKF4':
            info1 = 'R: ' + str(R)
            info2 = 'Q: ' + str(Q)
            info3 = 'Particles: ' + str(n_of_particles)
            info4 = 'Sigma P: ' + str(std_add)

        '''
        fillerx1 = np.reshape( (self.est_mean[0,:]-self.est_var[0,:]), (self.n,))
        fillerx2 = np.reshape( (self.est_mean[0,:]+self.est_var[0,:]), (self.n,)) 

        fillery1 = np.reshape( (self.est_mean[1,:]-self.est_var[1,:]), (self.n,))
        fillery2 = np.reshape( (self.est_mean[1,:]+self.est_var[1,:]), (self.n,))
        '''

        pl.figure(1)
        if method == 'PKF':
            pl.title(method +" 2D Pos[m] - " + info1 + " - " + info2 + " - " + info3 + " - " + info4)
        else:
            pl.title(method +" 2D Pos[m] - " + info1 + " - " + info2)
        pl.plot(self.est_mean[0,:], self.est_mean[1,:], label="est_pos(x,y)", color=quadcolor[2])
        #pl.fill_betweenx( self.est_mean[1,:], fillerx1, fillerx2, alpha=0.6, color=quadcolor[2])
        #pl.fill_between( self.est_mean[0,:], fillery1, fillery2, alpha=0.6, color=quadcolor[2])

        pl.plot(self.gt_mean[0,:], self.gt_mean[1, :], label="Ground Truth(x,y)", color=quadcolor[0])
        pl.xlabel("East")
        pl.ylabel("South")
        pl.legend()
        pl.savefig('results/'+method+'_2D_pos.png')
        
        filler1 = np.reshape( (self.Ed_mean[:]-self.Ed_var[:]), (self.n,))
        filler2 = np.reshape( (self.Ed_mean[:]+self.Ed_var[:]), (self.n,))

        pl.figure(2)
        if method == 'PKF':
            pl.title(method+" Error Dist[m] - " + info1 + " - " + info2 + " - " + info3 + " - " + info4+ "\n" + "Mean error(t=100->300): " + str(np.mean(self.Ed_mean[10000:30000])))
        else:
            pl.title(method+" Error Dist[m] - " + info1 + " - " + info2 + "\n" + "Mean error(t=100->300): " + str(np.mean(self.Ed_mean[10000:30000])))
        pl.plot(self.time, self.Ed_mean, label="Distance: est_pos - true_pos", color=quadcolor[2])
        pl.fill_between( self.time, filler1, filler2, alpha=0.6)
        pl.ylim(-0.1,3)
        pl.xlabel("Time [s]")
        pl.ylabel("Formation distance error [m]")
        pl.grid()
        pl.legend()
        pl.savefig('results/'+method+'_err_pos.png')
        
        filler1 = np.reshape( (self.est_mean[2,:]-self.est_var[2:,]), (self.n,))
        filler2 = np.reshape( (self.est_mean[2,:]+self.est_var[2:,]), (self.n,))

        pl.figure(3)
        if method == 'PKF':
            pl.title(method+" Altitude[m] - " + info1 + " - " + info2 + " - " + info3 + " - " + info4)
        else:
            pl.title(method+" Altitude[m] - " + info1 + " - " + info2)

        pl.plot(self.time, self.est_mean[2,:], label="est_alt", color=quadcolor[2])
        pl.fill_between( self.time, filler1, filler2, alpha=0.6)

        pl.plot(self.time, self.gt_mean[2,:], label="Ground Truth(alt)", color=quadcolor[0])
        pl.ylim(-5, 1)
        pl.xlabel("Time [s]")
        pl.ylabel("Altitude [m]")
        pl.grid()
        pl.legend(loc=2)
        pl.savefig('results/'+method+'_alt.png')

        #pl.pause(0)


if __name__ == "__main__":
    c_in = sys.argv[1]
    if c_in == 0:
        method_list = ['NF4', 'NF']
    elif c_in == 1:
        method_list = ['KF4', 'KF']
    elif c_in == 2:
        method_list = ['PF4', 'PF']
    elif c_in == 3:
        method_list = ['PKF4', 'PKF']

    #method_list = ['NF', 'NF4', 'KF4', 'KF', 'PF4', 'PF', 'PKF4', 'PKF']
    for method in method_list:
        l = logger(method)
        l.run_logger()
        l.calc_statistics()
        l.plot()
