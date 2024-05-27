import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from itertools import islice

def CWRU(item_path):
    axis = ["_DE_time", "_FE_time", "_BA_time"]
    datanumber = os.path.basename(item_path).split(".")[0]
    if eval(datanumber) < 100:
        realaxis = "X0" + datanumber + axis[0]
    else:
        realaxis = "X" + datanumber + axis[0]
    signal = loadmat(item_path)[realaxis]

    return signal

def JNU(item_path):
    fl = np.loadtxt(item_path)
    signal = fl.reshape(-1,1)
    
    return signal

def HUST(item_path):
    if 'sim' in item_path:
        signal = loadmat(item_path)['X_sim_with_noise']
    else:
        signal = loadmat(item_path)['data']
    return signal

def SEU(item_path):
    f = open(item_path, "r", encoding='gb18030', errors='ignore')
    fl = []
    if  "ball_20_0.csv" in item_path:
        for line in islice(f, 16, None):  # Skip the first 16 lines
            line = line.rstrip()
            word = line.split(",", 8)  # Separated by commas
            fl.append(eval(word[1]))  # Take a vibration signal in the x direction as input
    else:
        for line in islice(f, 16, None):  # Skip the first 16 lines
            line = line.rstrip()
            word = line.split("\t", 8)  # Separated by \t
            fl.append(eval(word[1]))  # Take a vibration signal in the x direction as input
    #--------------------
    fl = np.array(fl)
    signal = fl.reshape(-1, 1)
    
    return signal
    