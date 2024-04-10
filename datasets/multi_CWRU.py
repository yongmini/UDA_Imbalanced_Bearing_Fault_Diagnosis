import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm

#Digital data was collected at 12,000 samples per second
signal_size = 1024
dataname= {0:["97.mat","105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat","234.mat"],  # 1797rpm
           1:["98.mat","106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat","235.mat"],  # 1772rpm
           2:["99.mat","107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat","236.mat"],  # 1750rpm
           3:["100.mat","108.mat", "121.mat","133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat","237.mat"]}  # 1730rpm
#12kdrive       noraml     ir007        b007       or007(6)        ir014      b014          or014(6)        ir021        b021      or021(6)
 
label_mapping = {
    0: 0,  # normal
    1: 1, 2: 2, 3: 3,  
    4: 1, 5: 2, 6: 3,  
    7: 1, 8: 2, 9: 3   
}
datasetname = ["12k_Drive_End_Bearing_Fault_Data", "12k_Fan_End_Bearing_Fault_Data", "48k_Drive_End_Bearing_Fault_Data",
               "Normal_Baseline_Data"]

axis = ["_DE_time", "_FE_time", "_BA_time"]

label = [i for i in range(0, 10)]

def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab =[]
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            if n==0:
               path1 =os.path.join(root,datasetname[3], dataname[N[k]][n])
            else:
                path1 = os.path.join(root,datasetname[0], dataname[N[k]][n])
            data1, lab1 = data_load(path1,dataname[N[k]][n],label=label_mapping[label[n]])
            lab1=k*100+np.array(lab1)#k是域标签,lab1代表的是类标签
            lab1=lab1.tolist()
            data += data1
            lab +=lab1
    return [data, lab]


def data_load(filename, axisname, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
    else:
        realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[realaxis]
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab

def balance_data(data_pd):
    count = data_pd.value_counts(subset='label')
    min_len = min(count) - 1
    df = pd.DataFrame(columns=('data', 'label'))
    for i in count.keys():
        data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True)
        df = pd.concat([df, data_pd_tmp.loc[:min_len, ['data', 'label']]], ignore_index=True)
    return df
#--------------------------------------------------------------------------------------------------------------------
class multi_CWRU(object):
    num_classes = 10
    inputchannel = 1
    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir+"/CWRU"
        self.source_N = transfer_task[0]
        print('self.source_N',self.source_N)
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            
            print("Source Train Dataset Size:", len(source_train))
            print("Source Validation Dataset Size:", len(source_val))
            print("Target Train Dataset Size:", len(target_train))
            print("Target Validation Dataset Size:", len(target_val))

            # 레이블에 따른 샘플 수 출력
            print("Source Train Label Counts:")
            print(train_pd['label'].value_counts())
            print("Source Validation Label Counts:")
            print(val_pd['label'].value_counts())
            print("Target Train Label Counts:")
            print(train_pd['label'].value_counts())
            print("Target Validation Label Counts:")
            print(val_pd['label'].value_counts())
            
            return source_train, source_val, target_train, target_val
        else:
            #get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val


"""
    def data_split(self):

"""