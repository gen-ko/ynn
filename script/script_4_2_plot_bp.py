import src.plotter as plotter
import pickle

import numpy
import matplotlib.pyplot as plt
import os
from src import train as utf


f1 = '../output/metric/script-1-1-1-e16-i-s.dump'
f2 = '../output/metric/script-1-1-2-e16-i-s.dump'
f3 = '../output/metric/script-1-1-3-e16-i-s.dump'
with open(f1, 'rb') as f:
    s1, s2 = pickle.load(f)
    
with open(f2, 'rb') as f:
    s3, s4 = pickle.load(f)
    
with open(f3, 'rb') as f:
    s5, s6 = pickle.load(f)

v1 = s1.loss
v2 = s2.loss
v3 = s3.loss
v4 = s4.loss
v5 = s5.loss
v6 = s6.loss


plotter.plot_general([v1, v2, v3, v4, v5,  v6], ['v1', 'v2', 'v3', 'v4', 'v5', 'v6'], 'x', 'y', '../output/metric/fig1.png', 'title')