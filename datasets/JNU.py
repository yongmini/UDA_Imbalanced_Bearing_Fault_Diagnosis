import os
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm
from tabulate import tabulate
signal_size = 1024



#Three working conditions

dataname= {0:["ib600_2.csv","n600_3_2.csv","ob600_2.csv","tb600_2.csv"],
           1:["ib800_2.csv","n800_3_2.csv","ob800_2.csv","tb800_2.csv"],
           2:["ib1000_2.csv","n1000_3_2.csv","ob1000_2.csv","tb1000_2.csv"]}

label = [i for i in range(0,4)]


#generate Training Dataset and Testing Dataset
def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    
    data = []
    lab =[]
    for k in range(len(N)):
        for i in tqdm(range(len(dataname[N[k]]))):
            path1 = os.path.join('/tmp',root,dataname[N[k]][i])
            data1, lab1 = data_load(path1,label=label[i])
            data += data1
            lab +=lab1


    return [data, lab]

def data_load(filename,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = np.loadtxt(filename)
    fl = fl.reshape(-1,1)
    data=[] 
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start +=signal_size
        end +=signal_size

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
class JNU(object):
    num_classes = 4
    inputchannel = 1

    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir+"/JNU"
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
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            
            data_pd = balance_data(data_pd) 
            
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            
            # Table for source data distribution
            source_counts = pd.concat([train_pd['label'].value_counts(), val_pd['label'].value_counts()], axis=1, keys=['Train', 'Validation'])
            print(source_counts)
            source_counts.sort_index(inplace=True)
            source_counts.loc['Dataset Size'] = [len(source_train), len(source_val)]
            print("Source Data Distribution:")
            print(tabulate(source_counts, headers='keys', tablefmt='psql'))

            
            # get target data and split into train and val
            list_data = get_files(self.data_dir, self.target_N)
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