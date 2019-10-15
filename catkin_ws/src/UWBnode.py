#!/usr/bin/python
import rospy
from std_msgs.msg import Int16MultiArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import numpy as np
from matplotlib import animation
import math
import random
import sys
import time
'''
Data vector : [ id,  x_pos,  y_pos,  z_pos ]
'''


class UWBros:
    def __init__(self, id, NofUWBs, pos):
        self.ID = id
        self.NofUWBs = NofUWBs
        self.pos = pos
        self.pub_list = Int16MultiArray()
        self.pub_list.data = []

        print("Initializing UWB "+str(self.ID) + " with position: " + str(pos))
        rospy.init_node('UWB'+str(self.ID), anonymous=True)

        self.UWB_pub = rospy.Publisher("UWBdata", Int16MultiArray, queue_size=10)
        self.UWB_sub_list = []

        for i in range(int(self.NofUWBs)):
            if i != self.ID:
                self.UWB_sub_list.append(rospy.Subscriber("UWBdata", Int16MultiArray, self.callback))
        self.pub_list.data.append(self.ID)
        for i in range(3):
            self.pub_list.data.append(pos[i])

        rospy.Timer(rospy.Duration(1./0.5), self.timer_callback)


    def callback(self, data_in):
        data = data_in.data
        if data[0] != self.ID:
            print("UWB node " + str(self.ID) + " prints:   Distance between " + str(self.ID) + " and " + str(data[0]) + " is: ")
            dis1 = math.sqrt(self.pos[0]**2 + self.pos[1]**2 + self.pos[2]**2)
            dis2 = math.sqrt(data[1]**2 + data[2]**2 + data[3]**2)
            print(abs(dis1-dis2))

    def timer_callback(self, event):
        self.UWB_pub.publish(self.pub_list)

if __name__ == '__main__':
    NofUWBs = 3
    pos_list = [[0,0,0],[3,4,0],[2,7,5]]
    uwb_node_list = list()

    for i in range(NofUWBs):
        uwb_node_list.append( UWBros(id=i, NofUWBs=NofUWBs, pos=pos_list[i]) )


    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
