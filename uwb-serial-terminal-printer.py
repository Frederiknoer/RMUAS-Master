#!/usr/bin/env python

import serial
import csv

baud = 115200

ser = serial.Serial('/dev/ttyUSB0', baud)

while True:
    data = ser.readline()
    print(data)
    #if data:
        #print(data.decode("utf-8"))
