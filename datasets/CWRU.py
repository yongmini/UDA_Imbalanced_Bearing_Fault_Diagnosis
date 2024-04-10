import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm
from tabulate import tabulate
#Digital data was collected at 12,000 samples per second
signal_size = 2048

# 기존 
dataname= {0:["97.mat","105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat","234.mat"],  # 1797rpm
           1:["98.mat","106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat","235.mat"],  # 1772rpm
           2:["99.mat","107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat","236.mat"],  # 1750rpm
           3:["100.mat","108.mat", "121.mat","133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat","237.mat"]}  # 1730rpm
#12kdrive       noraml     ir007        b007    or007(6)   ir014      b014      or014(6)    ir021      b021    or021(6)

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

def get_files(root, N,source=True):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab =[]
    for k in range(len(N)):
        if source:
            
            print("Loading data from source dataset: ", N[k])
            
        else:
            print("Loading data from target dataset: ", N[k])    
        for n in tqdm(range(len(dataname[N[k]]))):
            if n==0:
               path1 =os.path.join(root,datasetname[3], dataname[N[k]][n])
            else:
                path1 = os.path.join(root,datasetname[0], dataname[N[k]][n])
            data1, lab1 = data_load(path1,dataname[N[k]][n],label=label_mapping[label[n]])
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
class CWRU(object):
    num_classes = 4
    inputchannel = 1
    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir+"/CWRU"
        self.source_N = transfer_task[0]
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


    def data_split(self, transfer_learning=True, imbalance_ratio=None):
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N,source=True)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            
            data_pd = balance_data(data_pd) 
            
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            
            # Table for source data distribution
            source_counts = pd.concat([train_pd['label'].value_counts(), val_pd['label'].value_counts()], axis=1, keys=['Train', 'Validation'])
            source_counts.sort_index(inplace=True)
            source_counts.loc['Dataset Size'] = [len(source_train), len(source_val)]
            print("Source Data Distribution:")
            print(tabulate(source_counts, headers='keys', tablefmt='psql'))

            
            # get target data and split into train and val
            list_data = get_files(self.data_dir, self.target_N,source=False)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
                        
            data_pd = balance_data(data_pd) 
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])

            # apply imbalance_ratio to target_train
            if imbalance_ratio is not None:
                train_data = []
                train_labels = []
                for label, ratio in imbalance_ratio.items():
                    if label == 0:  # Assume label 0 is normal class
                        train_data += train_pd[train_pd["label"] == label]["data"].tolist()
                        train_labels += train_pd[train_pd["label"] == label]["label"].tolist()
                    else:
                        num_samples = int(len(train_pd[train_pd["label"] == label]) * ratio)
                        train_data += train_pd[train_pd["label"] == label]["data"].tolist()[:num_samples]
                        train_labels += train_pd[train_pd["label"] == label]["label"].tolist()[:num_samples]
                train_pd = pd.DataFrame({"data": train_data, "label": train_labels})

            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

           # Print target data distribution
            target_counts = pd.concat([train_pd['label'].value_counts(), val_pd['label'].value_counts()], axis=1, keys=['Train', 'Validation'])
            target_counts.sort_index(inplace=True)
            target_counts.loc['Dataset Size'] = [len(target_train), len(target_val)]
            print("Target Data Distribution:")
            print(tabulate(target_counts, headers='keys', tablefmt='psql'))

            return source_train, source_val, target_train, target_val
        else:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            data_pd = balance_data(data_pd) 
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            data_pd = balance_data(data_pd) 
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])

            return source_train, source_val, target_val



"""
    def data_split(self):

"""