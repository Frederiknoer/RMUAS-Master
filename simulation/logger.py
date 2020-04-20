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


n_of_sims = 1


class logger:
    def __init__(self):
        self.N = n_of_sims
        tf = 400
        dt = 1/200
        self.time = np.linspace(0, tf, int(tf/dt))

        self.run_animation = False
        self.pos_method = 'PF' #NF = No filter, KF = kalman filter, PF = particle filter

        self.pycopter = pycopter_class.pycopter(tf, dt)

        n = int(tf/dt)

        #LOGS:
        self.big_log_Ed  = np.empty((0,1))
        self.big_log_est = np.empty((0,3))
        self.big_log_gt  = np.empty((0,3))


    def run_logger(self):
        for i in range(self.N):
            self.pycopter.run(method=self.pos_method, run_animation=self.run_animation) #temp_ed[i], temp_est[i], temp_gt[i] = 
            #self.big_log_Ed  = np.append(self.big_log_Ed,  temp_ed)
            #self.big_log_est = np.append(self.big_log_est, temp_est)
            #self.big_log_gt  = np.append(self.big_log_gt,  temp_gt)



    
    def calc_statistics(self):
        #Calculate statistics:
        self.Ed_mean  = np.mean(self.big_log_Ed, axis=0)
        self.Ed_var   = np.var(self.big_log_Ed, axis=0)

        self.est_mean = np.mean(self.big_log_est, axis=0)
        self.est_var  = np.var(self.big_log_est, axis=0)

        self.gt_mean  = np.mean(self.big_log_gt, axis=0)
        self.gt_var   = np.var(self.big_log_gt, axis=0)

    '''
    def parse_data(self):
        self.Ed_mean  = self.big_log_Ed[0][:]
        self.alt_mean = self.big_log_alt[0][:]
        self.x_mean   = self.big_log_x[0][:]
        self.y_mean   = self.big_log_y[0][:]
    '''

    def plot(self):
        quadcolor = ['r', 'g', 'b']
        pl.close("all")
        pl.ion()
        
        pl.figure(1)
        pl.title("2D Position [m]")
        pl.plot(self.big_log_est[0], self.big_log_est[1], label="mse+kf", color=quadcolor[2])
        pl.plot(self.big_log_gt[0], self.big_log_gt[1], label="gt", color=quadcolor[0])
        #pl.fill_between(x_mean + x_var/2, y_mean, label="mse+kf", color=quadcolor[2])
        pl.xlabel("East")
        pl.ylabel("South")
        pl.legend()
        pl.savefig("plots/2d_pos.png")

        pl.figure(2)
        pl.title("Error Distance [m]")
        pl.plot(self.time, self.big_log_Ed, label="mse+kf", color=quadcolor[2])
        #if n_of_sims > 1:
            #pl.fill_between(self.time, self.Ed_mean-self.Ed_var/2, self.Ed_mean+self.Ed_var/2, label="mse+kf", color=quadcolor[2], alpha=0.4)
        pl.xlabel("Time [s]")
        pl.ylabel("Formation distance error [m]")
        pl.grid()
        pl.legend()
        pl.savefig("plots/error.png")
        
        pl.figure(3)
        pl.title("Altitude Over Time")
        pl.plot(self.time, self.big_log_est[2], label="mse+kf", color=quadcolor[2])
        #if n_of_sims > 1:
            #pl.fill_between(self.time, self.z_mean-self.z_mean/2, self.z_mean+self.z_mean/2, label="mse+kf", color=quadcolor[2], alpha=0.4)
        pl.plot(self.time, self.big_log_gt[2], label="gt", color=quadcolor[0])
        pl.xlabel("Time [s]")
        pl.ylabel("Altitude [m]")
        pl.grid()
        pl.savefig("plots/alt.png")
        
        pl.pause(0)


if __name__ == "__main__":
    l = logger()
    l.run_logger()

    #l.calc_statistics()
    #l.plot()
