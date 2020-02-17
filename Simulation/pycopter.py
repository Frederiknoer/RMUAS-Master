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
    mu, sigma = 0, 0.1
    std_err = np.random.normal(mu, sigma, 1)[0]
    return (np.linalg.norm(p1 - p2)) + std_err

def vecLen(v):
    return math.sqrt(np.dot(v,v))

def getAngle(v1, v2):
    return np.arctan( (np.dot(v1, v2)) / (vecLen(v1)*vecLen(v2)) )

def getRotMat(q1, q2, q3, A, B, C):
    v1q = abs(q1 - q2)
    v1a = abs(A - B)
    v2q = abs(q1 - q3)
    v2a = abs(A - C)
    v3q = abs(q2 - q3)
    v3a = abs(B - C)

    a = (getAngle(v1q, v1a) + getAngle(v2q, v2a) + getAngle(v3q, v3a)) / 3
    #a = np.array([getAngle(v1q, v1a), getAngle(v2q, v2a), getAngle(v3q, v3a)])

    m11 = np.cos(a)
    m12 = np.sin(a)
    m21 = -np.sin(a)
    m22 = np.cos(a)

    return np.array([ [m11, m12],
                       [m21, m22]])

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

xyz0_0 = np.array([0.0, 0.0, 0.0])
xyz1_0 = np.array([3.0, 0.0, 0.0])
xyz2_0 = np.array([1.5, 1.75, 0.0])
#xyz3_0 = np.array([2.0, 0.0, 0.0])
xyz_uav_0 = np.array([2.0, 1.5, 0.0])

v_ned_0 = np.array([0.0, 0.0, 0.0])
w_0 = np.array([0.0, 0.0, 0.0, 0.0])

# Setting quads
uwb0 = quad.quadrotor(0, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz0_0, v_ned_0, w_0)

uwb1 = quad.quadrotor(1, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz1_0, v_ned_0, w_0)

uwb2 = quad.quadrotor(2, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz2_0, v_ned_0, w_0)

#uwb3 = quad.quadrotor(3, m, l, J, CDl, CDr, kt, km, kw, \
#        att_0, pqr_0, xyz3_0, v_ned_0, w_0)

UAV = quad.quadrotor(10, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz_uav_0, v_ned_0, w_0)

# Simulation parameters
tf = 800
dt = 5e-2
time = np.linspace(0, tf, tf/dt)
it = 0
frames = 100

# Data log
uwb0_log = quadlog.quadlog(time)
uwb1_log = quadlog.quadlog(time)
uwb2_log = quadlog.quadlog(time)
uwb3_log = quadlog.quadlog(time)
UAV_log = quadlog.quadlog(time)
Ed_log = np.zeros((time.size, 3))

# Plots
quadcolor = ['r', 'g', 'b']
pl.close("all")
pl.ion()
fig = pl.figure(0)
axis3d = fig.add_subplot(111, projection='3d')

init_area = 10
s = 2

# Desired altitude and heading
alt_d = 0
uwb0.yaw_d = -np.pi
uwb1.yaw_d =  np.pi/2
uwb2.yaw_d =  0
#uwb3.yaw_d = 0

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
#RA3 = range_agent.uwb_agent( ID=3 )

UAV_agent = range_agent.uwb_agent( ID=10 )

est_pos_kf = np.array([])

for t in time:
    if it % 50 == 0:
        UAV_agent.handle_range_msg(Id=RA0.id, range=get_dist(UAV.xyz, uwb0.xyz))
        #UAV_agent.handle_range_msg(Id=RA1.id, range=get_dist(UAV.xyz, uwb1.xyz))
        #UAV_agent.handle_range_msg(Id=RA2.id, range=get_dist(UAV.xyz, uwb2.xyz))
        #UAV_agent.handle_range_msg(Id=RA3.id, range=get_dist(UAV.xyz, uwb3.xyz))

        #UAV_agent.handle_other_msg(Id1=RA0.id, Id2=RA1.id, range=get_dist(uwb0.xyz, uwb1.xyz))
        #UAV_agent.handle_other_msg(Id1=RA0.id, Id2=RA2.id, range=get_dist(uwb0.xyz, uwb2.xyz))
        #UAV_agent.handle_other_msg(Id1=RA0.id, Id2=RA3.id, range=get_dist(uwb0.xyz, uwb3.xyz))

        #UAV_agent.handle_other_msg(Id1=RA1.id, Id2=RA2.id, range=get_dist(uwb1.xyz, uwb2.xyz))
        #UAV_agent.handle_other_msg(Id1=RA1.id, Id2=RA3.id, range=get_dist(uwb1.xyz, uwb3.xyz))

        #UAV_agent.handle_other_msg(Id1=RA2.id, Id2=RA3.id, range=get_dist(uwb2.xyz, uwb3.xyz))

        #est_pos = UAV_agent.calc_pos_MSE()

    if np.linalg.norm(UAV.acc) != 0:
        est_pos_kf = UAV_agent.handle_acc_msg(UAV.acc)
    print("True Pos: ", UAV.xyz)
    print("Estimated Pos: ", est_pos_kf[0:3])

    UAV.set_v_2D_alt_lya([random.uniform(-1.0,1.0), random.uniform(-1.0,1.0)], -5)


    #uwb0.step(dt)
    #uwb1.step(dt)
    #uwb2.step(dt)
    #uwb3.step(dt)
    UAV.step(dt)

    # Animation
    if it%frames == 0:
        pl.figure(0)
        axis3d.cla()
        ani.draw3d(axis3d, uwb0.xyz, uwb0.Rot_bn(), quadcolor[0])
        ani.draw3d(axis3d, uwb1.xyz, uwb1.Rot_bn(), quadcolor[0])
        ani.draw3d(axis3d, uwb2.xyz, uwb2.Rot_bn(), quadcolor[0])
        #ani.draw3d(axis3d, uwb3.xyz, uwb3.Rot_bn(), quadcolor[0])
        ani.draw3d(axis3d, UAV.xyz, UAV.Rot_bn(), quadcolor[1])
        axis3d.set_xlim(-5, 5)
        axis3d.set_ylim(-5, 5)
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


    # Log
    uwb0_log.xyz_h[it, :] = uwb0.xyz
    uwb0_log.att_h[it, :] = uwb0.att
    uwb0_log.w_h[it, :] = uwb0.w
    uwb0_log.v_ned_h[it, :] = uwb0.v_ned

    uwb1_log.xyz_h[it, :] = uwb1.xyz
    uwb1_log.att_h[it, :] = uwb1.att
    uwb1_log.w_h[it, :] = uwb1.w
    uwb1_log.v_ned_h[it, :] = uwb1.v_ned

    uwb2_log.xyz_h[it, :] = uwb2.xyz
    uwb2_log.att_h[it, :] = uwb2.att
    uwb2_log.w_h[it, :] = uwb2.w
    uwb2_log.v_ned_h[it, :] = uwb2.v_ned

    #uwb3_log.xyz_h[it, :] = uwb3.xyz
    #uwb3_log.att_h[it, :] = uwb3.att
    #uwb3_log.w_h[it, :] = uwb3.w
    #uwb3_log.v_ned_h[it, :] = uwb3.v_ned

    UAV_log.xyz_h[it, :] = UAV.xyz
    UAV_log.att_h[it, :] = UAV.att
    UAV_log.w_h[it, :] = UAV.w
    UAV_log.v_ned_h[it, :] = UAV.v_ned

    it+=1

    # Stop if crash
    if (uwb0.crashed == 1 or uwb1.crashed == 1 or uwb2.crashed == 1 or  UAV.crashed == 1):
        break

pl.figure(1)
pl.title("2D Position [m]")
pl.plot(q1_log.xyz_h[:, 0], q1_log.xyz_h[:, 1], label="q1", color=quadcolor[0])
pl.plot(q2_log.xyz_h[:, 0], q2_log.xyz_h[:, 1], label="q2", color=quadcolor[1])
pl.plot(q3_log.xyz_h[:, 0], q3_log.xyz_h[:, 1], label="q3", color=quadcolor[2])
pl.xlabel("East")
pl.ylabel("South")
pl.legend()

pl.figure(2)
pl.plot(time, q1_log.att_h[:, 2], label="yaw q1")
pl.plot(time, q2_log.att_h[:, 2], label="yaw q2")
pl.plot(time, q3_log.att_h[:, 2], label="yaw q3")
pl.xlabel("Time [s]")
pl.ylabel("Yaw [rad]")
pl.grid()
pl.legend()

pl.figure(3)
pl.plot(time, -q1_log.xyz_h[:, 2], label="$q_1$")
pl.plot(time, -q2_log.xyz_h[:, 2], label="$q_2$")
pl.plot(time, -q3_log.xyz_h[:, 2], label="$q_3$")
pl.xlabel("Time [s]")
pl.ylabel("Altitude [m]")
pl.grid()
pl.legend(loc=2)

pl.figure(4)
pl.plot(time, Ed_log[:, 0], label="$e_1$")
pl.plot(time, Ed_log[:, 1], label="$e_2$")
pl.plot(time, Ed_log[:, 2], label="$e_3$")
pl.xlabel("Time [s]")
pl.ylabel("Formation distance error [m]")
pl.grid()
pl.legend()

pl.pause(0)
