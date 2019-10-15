#!/usr/bin/env python

import localization as lx
import serial
import csv
import matplotlib.pyplot as plt

def get_uwb_id(data):
    id_idx = data.find('from:')
    if id_idx == -1:
        return None
    id = ''
    while data[id_idx+6] != ' ':
        id += data[id_idx+6]
        id_idx += 1
    return id

def get_uwb_range(data):
    range_idx = data.find('Range:')
    if range_idx == -1:
        return None
    data_val = (float(data[range_idx+7])) + (float(data[range_idx+9])*0.1) + (float(data[range_idx+10])*0.01)
    return data_val

ser = serial.Serial('/dev/ttyUSB0', 115200)

P=lx.Project(mode='3D',solver='LSE')

P.add_anchor('anchore_A',(0,100,50))
P.add_anchor('anchore_B',(100,100,25))
P.add_anchor('anchore_C',(100,0,15))

t,label=P.add_target()

t.add_measure('anchore_A',50)
t.add_measure('anchore_B',50)
t.add_measure('anchore_C',50)

P.solve()

# Then the target location is:

print(t.loc)
