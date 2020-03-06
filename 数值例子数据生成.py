# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 18:05:31 2020

@author: SunShine
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

np.random.seed(42)

x1 = np.random.normal(1,1,3000).reshape(-1,1)
x2 = np.random.normal(1,0.8,3000).reshape(-1,1)
x3 = np.random.normal(0.7,1,3000).reshape(-1,1)
x4 = np.random.normal(0.5,0.8,3000).reshape(-1,1)
x5 = np.random.normal(0.3,0.7,3000).reshape(-1,1)
x6 = np.random.normal(0.,1.0,3000).reshape(-1,1)
x7 = ((x1+x2)/2).reshape(-1,1)
x8 = ((x3+x4)/2).reshape(-1,1)
x9 = ((x5+x6)/2).reshape(-1,1)
x10 = ((x1+x4)/2).reshape(-1,1)
x11 = ((x2+x5)/2).reshape(-1,1)
x12 = ((x3+x6)/2).reshape(-1,1)
x13 = ((x1+x2+x3+x4+x5+x6)/6).reshape(-1,1)

y = [x1[i]**2 + math.sin(x2[i])+ math.cos(x3[i])+ np.exp(x4[i])+ \
    x5[i]**2 + math.sin(x6[i])+ math.cos(x7[i])+ np.exp(x8[i])+ \
    x9[i]**2 + math.sin(x10[i])+ math.cos(x11[i])+ np.exp(x12[i])+ x13[i]**3 for i in range(3000)]
miny = min(y)
maxy = max(y)
y = (y-miny)/(maxy - miny)
Data_set = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,y),axis=1)

np.savetxt('.\data.csv', Data_set, delimiter = ',')
#data = np.loadtxt(open(".\data.csv","rb"),delimiter=",",skiprows=0)
#plt.figure()
#plt.plot(data[:,13])
#plt.show()