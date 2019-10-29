#!/usr/bin/env python

import serial
import csv

ser = serial.Serial('/dev/ttyUSB0', 115200)

while True:
    data = ser.readline()
    if data:
        print(data)
