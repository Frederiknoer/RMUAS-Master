#!/usr/bin/env python
import serial
import csv
import numpy as np
import matplotlib.pyplot as plt

#************************************************************ SET NAME AND FOLDER HERE ***************************************

folder = "m4/"
name = "200cm"


#*****************************************************************************************************************************



def get_uwb_id(data):
    id_idx = data.find('from:')
    if id_idx == -1:
        return None
    id = ''
    while data[id_idx+6] != ' ':
        id += data[id_idx+6]
        id_idx += 1
    id = id[:-1]
    return id

def get_uwb_range(data):
    range_idx = data.find('Range:')
    if range_idx == -1:
        return None
    data_val = (float(data[range_idx+7])) + (float(data[range_idx+9])*0.1) + (float(data[range_idx+10])*0.01)
    return data_val

ser = serial.Serial('/dev/ttyUSB0', 115200)
x = []

while len(x) < 250:
    data = ser.readline()
    if data:
        id = get_uwb_id(data)
        range = get_uwb_range(data)
        if id == None or range == None:
            continue
        x.append(float(range))
        with open('distance_tests/csv/'+folder+name+'.csv', mode='a') as writeFile:
            writer = csv.writer(writeFile, delimiter=',')
            writer.writerow([range])
        writeFile.close()

fig, ax = plt.subplots()
ax.set_title(name)
ax.boxplot(x, showfliers=False)
#plt.show()
plt.savefig('distance_tests/plots/'+folder+name+'.png')
