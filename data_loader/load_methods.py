from scipy.io import loadmat
import numpy as np
import os
from itertools import islice

def PU(item_path):
    name = os.path.basename(item_path).split(".")[0]
    fl = loadmat(item_path)[name]
    signal = fl[0][0][2][0][6][2] 
    signal = signal.reshape(-1,1)

    return signal
def HUST(item_path):
    signal = loadmat(item_path)['data']
    return signal


def CWRU(item_path):
    axis = ["_DE_time", "_FE_time", "_BA_time"]
    datanumber = os.path.basename(item_path).split(".")[0].split("_")[0] 
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

def MFPT(item_path):
    f = item_path.split("/")[-2]
    if f == 'normal':
        signal = (loadmat(item_path)["bearing"][0][0][1])
    else:
        signal = (loadmat(item_path)["bearing"][0][0][2])

    return signal

def SEU(item_path):

    dataname = os.path.basename(item_path)
    
    with open(item_path, "r", encoding='gb18030', errors='ignore') as f:
        fl = []
        if dataname == "ball_20_0.csv":
            print("yes")
            for line in islice(f, 16, None): 
                line = line.rstrip()
                word = line.split(",", 8)   
                fl.append(float(word[1]))  
        else:
            for line in islice(f, 16, None): 
                line = line.rstrip()
                word = line.split("\t", 8)  
                fl.append(float(word[1]))  
    
    signal = np.array(fl).reshape(-1, 1)
    return signal