#!/usr/bin/env python

import math
import numpy as np
import sympy as symp
import scipy as scip
import csv
import matplotlib.mlab as mlab
import matplotlib.pyplot as pl

plot_name_list = ['0.5m', '0.75m', '1.0m', '1.5m', '2.0m', '2.5m', '3.0m', '4.0m']
a_all = np.array([])
a = np.empty(250)

for plot_file in plot_name_list:
    np.delete(a, 0)
    a = np.genfromtxt('distance_tests/csv/m7/'+plot_file+'.csv', delimiter=',')

    mu = a.mean()
    #sigma = a.std()
    a_all = np.append(a_all, a-mu)

#bin_arr = [mu-0.06,mu-0.05,mu-0.04,mu-0.03,mu-0.02,mu-0.01,mu,mu+0.01,mu+0.02,mu+0.03,mu+0.04,mu+0.05,mu+0.06]

mean = 0
sigma = a_all.std()
print mean
print sigma

pl.figure()

bin_arr = [-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0,0.01,0.02,0.03,0.04,0.05,0.06]

ln = np.linspace(-3 * sigma + mean, 3 * sigma + mean, len(bin_arr))
print(ln)

np.histogram(a_all,bins = bin_arr) 
hist,bins = np.histogram(a_all,bins = bin_arr) 

pl.hist(a_all,bins = bin_arr, edgecolor='black', linewidth=1.2) 
pl.plot(ln, bin_arr)
pl.title('Combined Data, with subtracted mean')

pl.savefig('distance_tests/plots/hists/combined.png')
