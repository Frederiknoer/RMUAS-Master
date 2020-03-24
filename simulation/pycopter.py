from scipy import linalg as la
import matplotlib.pyplot as pl
import numpy as np
import math
import sys
import random

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import uwb_agent as range_agent

sys.path.append("pycopter/")
import quadrotor as quad
import formation_distance as form
import quadlog
import animation as ani

PI = 3.14159265359

def get_dist_clean(p1, p2):
    return (np.linalg.norm(p1 - p2))

def get_dist(p1, p2):
    mu, sigma = 0, 0.01
    std_err = np.random.normal(mu, sigma, 1)[0]
    return (np.linalg.norm(p1 - p2)) + std_err

# Quadrotor
m = 0.65 # Kg
l = 0.23 # m
Jxx = 7.5e-3 # Kg/m^2
Jyy = Jxx
Jzz = 1.3e-2
Jxy = 0
Jxz = 0
Jyz = 0
J = np.array([[Jxx, Jxy, Jxz], \
              [Jxy, Jyy, Jyz], \
              [Jxz, Jyz, Jzz]])
CDl = 9e-3
CDr = 9e-4
kt = 3.13e-5  # Ns^2
km = 7.5e-7   # Ns^2
kw = 1/0.18   # rad/s

# Initial conditions
att_0 = np.array([0.0, 0.0, 0.0])
pqr_0 = np.array([0.0, 0.0, 0.0])

d = 4.0
dy = d * (np.sqrt(3)/2)

xyz0_0 = np.array([0.0, 0.0, 0.0])
xyz1_0 = np.array([d,   0.0, 0.0])
xyz2_0 = np.array([d/2, dy, 0.0])
xyz3_0 = np.array([d/2, -dy, 0.1])

xyz4_0 = np.array([-(d/2), dy, 0.0])
xyz5_0 = np.array([-d, 0.0, 0.0])
xyz6_0 = np.array([-(d/2), -dy, 0.0])

xyz_uav_0 = np.array([1.0, 1.5, 0.0])


state = 0
#wp = np.array([ [ 2,  2, -5 ], [ 2, -2, -5], [ -2, -2, -5 ], [-2,  2, -5] ])
#wp = np.array([ [ d,  dy+1, -3 ], [ d, -(dy+1), -3 ], [ -1, -(dy+1), -3 ], [-1,  dy+1, -3 ] ])
wp = np.array([ [ d,  dy+1, -3 ], [ d, -dy+1, -3 ], [ -1, -dy-1, -3 ], [-1,  dy+1, -3 ] ])


v_ned_0 = np.array([0.0, 0.0, 0.0])
w_0 = np.array([0.0, 0.0, 0.0, 0.0])

# Setting quads
uwb0 = quad.quadrotor(0, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz0_0, v_ned_0, w_0)

uwb1 = quad.quadrotor(1, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz1_0, v_ned_0, w_0)

uwb2 = quad.quadrotor(2, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz2_0, v_ned_0, w_0)

uwb3 = quad.quadrotor(3, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz3_0, v_ned_0, w_0)
'''
uwb4 = quad.quadrotor(4, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz4_0, v_ned_0, w_0)

uwb5 = quad.quadrotor(5, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz5_0, v_ned_0, w_0)

uwb6 = quad.quadrotor(6, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz6_0, v_ned_0, w_0)
'''
UAV = quad.quadrotor(10, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz_uav_0, v_ned_0, w_0)

# Simulation parameters
tf = 500
dt = 0.2
time = np.linspace(0, tf, tf/dt)
it = 0
frames = 50

# Data log
kf_log = quadlog.quadlog(time)
mse_log = quadlog.quadlog(time)
ls_log = quadlog.quadlog(time)
UAV_log = quadlog.quadlog(time)
Ed_log = np.zeros((time.size, 1))
eig_log = np.zeros((time.size, 3))

# Plots
quadcolor = ['r', 'g', 'b']
pl.close("all")
pl.ion()
fig = pl.figure(0)
axis3d = fig.add_subplot(111, projection='3d')

init_area = 10
s = 2

'''
Position Rules:
ID = 0: Origin of the local coordinate system (a)
ID = 1: Always 0 in y- and z-components (b)
ID = 2: Always posetive in both x- and y-components

ID = 10: Always the UAV
'''

RA0 = range_agent.uwb_agent( ID=0 )
RA1 = range_agent.uwb_agent( ID=1 )
RA2 = range_agent.uwb_agent( ID=2 )
RA3 = range_agent.uwb_agent( ID=3 )
RA4 = range_agent.uwb_agent( ID=4 )
RA5 = range_agent.uwb_agent( ID=5 )
RA6 = range_agent.uwb_agent( ID=6 )

UAV_agent = range_agent.uwb_agent( ID=10, d=d)
kalmanStarted = False
kalmanStarted_int = 0


for t in time:
    if it % 5 == 0 or it == 0:
        UAV_agent.handle_range_msg(Id=RA0.id, range=get_dist(UAV.xyz, uwb0.xyz))
        UAV_agent.handle_range_msg(Id=RA1.id, range=get_dist(UAV.xyz, uwb1.xyz))
        UAV_agent.handle_range_msg(Id=RA2.id, range=get_dist(UAV.xyz, uwb2.xyz))
        UAV_agent.handle_range_msg(Id=RA3.id, range=get_dist(UAV.xyz, uwb3.xyz))
        #UAV_agent.handle_range_msg(Id=RA4.id, range=get_dist(UAV.xyz, uwb4.xyz))
        #UAV_agent.handle_range_msg(Id=RA5.id, range=get_dist(UAV.xyz, uwb5.xyz))
        #UAV_agent.handle_range_msg(Id=RA6.id, range=get_dist(UAV.xyz, uwb6.xyz))

    if UAV.xyz[2] < -2 and not kalmanStarted:
        UAV_agent.startKF(UAV.xyz, UAV.acc)
        kalmanStarted = True
        kalmanStarted_int = 1
    
    if kalmanStarted:
        UAV_agent.handle_acc_msg(acc_in=UAV.acc)

    mse_pos = UAV_agent.calc_pos_MSE()
    mse_log.xyz_h[it, :] = mse_pos
    Ed_log[it, :] = np.array([ get_dist_clean(mse_pos, UAV.xyz) ])

    x_err = abs(wp[state][0] - UAV.xyz[0])
    y_err = abs(wp[state][1] - UAV.xyz[1])

    if state == 0:
        UAV.set_v_2D_alt_lya(np.array([x_err*0.02, y_err*0.02]), -3)
        if get_dist_clean(UAV.xyz, wp[state]) < 0.4:
            state = 2
    elif state == 1:
        UAV.set_v_2D_alt_lya(np.array([x_err*0.02, -y_err*0.02]),-3)
        if get_dist_clean(UAV.xyz, wp[state]) < 0.4:
            state = 0
    elif state == 2:
        UAV.set_v_2D_alt_lya(np.array([-x_err*0.02, -y_err*0.02]), -3)
        if get_dist_clean(UAV.xyz, wp[state]) < 0.4:
            state = 3
    elif state == 3:
        UAV.set_v_2D_alt_lya(np.array([-x_err*0.02, y_err*0.02]), -3)
        if get_dist_clean(UAV.xyz, wp[state]) < 0.4:
            state = 1

    UAV.step(dt)
    
    # Animation
    if it%frames == 0:
        pl.figure(0)
        axis3d.cla()
        ani.draw3d(axis3d, uwb0.xyz, uwb0.Rot_bn(), quadcolor[2])

        ani.draw3d(axis3d, uwb1.xyz, uwb1.Rot_bn(), quadcolor[0])
        ani.draw3d(axis3d, uwb2.xyz, uwb2.Rot_bn(), quadcolor[0])
        ani.draw3d(axis3d, uwb3.xyz, uwb3.Rot_bn(), quadcolor[0])

        #ani.draw3d(axis3d, uwb4.xyz, uwb4.Rot_bn(), quadcolor[0])
        #ani.draw3d(axis3d, uwb5.xyz, uwb5.Rot_bn(), quadcolor[0])
        #ani.draw3d(axis3d, uwb6.xyz, uwb6.Rot_bn(), quadcolor[0])

        ani.draw3d(axis3d, UAV.xyz, UAV.Rot_bn(), quadcolor[1])
        axis3d.set_xlim(-5, 5)
        axis3d.set_ylim(-1, 5)
        axis3d.set_zlim(0, 10)
        axis3d.set_xlabel('South [m]')
        axis3d.set_ylabel('East [m]')
        axis3d.set_zlabel('Up [m]')
        axis3d.set_title("Time %.3f s" %t)
        pl.pause(0.001)
        pl.draw()
        #namepic = '%i'%it
        #digits = len(str(it))
        #for j in range(0, 5-digits):
        #    namepic = '0' + namepic
        #pl.savefig("./images/%s.png"%namepic)
    

    UAV_log.xyz_h[it, :] = UAV.xyz
    UAV_log.att_h[it, :] = UAV.att
    UAV_log.w_h[it, :] = UAV.w
    UAV_log.v_ned_h[it, :] = UAV.v_ned

    it+=1

    # Stop if crash
    if (UAV.crashed == 1):
        break


pl.figure(1)
pl.title("2D Position [m]")
pl.plot(mse_log.xyz_h[:, 0],mse_log.xyz_h[:, 1], label="mse+kf", color=quadcolor[2])
pl.plot(UAV_log.xyz_h[:, 0], UAV_log.xyz_h[:, 1], label="Ground Truth", color=quadcolor[0])
pl.xlabel("East")
pl.ylabel("South")
pl.legend()


pl.figure(2)
pl.title("Error Distance [m]")
pl.plot(time, Ed_log[:, 0], label="mse+kf", color=quadcolor[2])
pl.xlabel("Time [s]")
pl.ylabel("Formation distance error [m]")
pl.grid()
pl.legend()


pl.figure(3)
pl.title("Altitude Over Time")
pl.plot(time, mse_log.xyz_h[:, 2], label="mse+kf", color=quadcolor[2])
pl.plot(time, UAV_log.xyz_h[:, 2], label="Ground Truth", color=quadcolor[0])
pl.xlabel("Time [s]")
pl.ylabel("Altitude [m]")
pl.grid()
pl.legend(loc=2)
'''
if METHOD == 'KF':
    pl.figure(4)
    pl.title("Eigen Covariance")
    pl.plot(time, eig_log[:, 0], label="cov(x,x)", color=quadcolor[2])
    pl.plot(time, eig_log[:, 1], label="cov(y,y)", color=quadcolor[1])
    pl.plot(time, eig_log[:, 2], label="cov(z,z)", color=quadcolor[0])
    #pl.plot(time, Ed_log[:, 3], label="$e_4$")
    #pl.plot(time, Ed_log[:, 4], label="$e_5$")
    pl.xlabel("Time [s]")
    pl.ylabel("[m]")
    pl.grid()
    pl.legend()
'''


pl.pause(0)
