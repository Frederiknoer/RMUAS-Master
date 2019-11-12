#!/usr/bin/env python

import serial
import csv

baud = 9600 #115200

ser = serial.Serial('/dev/ttyUSB0', baud)

while True:
    data = ser.readline()
    if data:
        print(data)
