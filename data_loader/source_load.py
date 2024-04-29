import os
import importlib
import pandas as pd
from scipy.io import loadmat
from tabulate import tabulate
import aug
import data_utils
import load_methods

def get_files(root, dataset, faults, signal_size, condition=3):
    data, labels = [], []

    data_load = getattr(load_methods, dataset)
    
    for index, name in enumerate(faults):
        data_dir = os.path.join(root, f'condition_{condition}', name)
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            signal = data_load(item_path)
            
            start, end = 0, signal_size
            while end <= signal.shape[0]:
                data.append(signal[start:end])
                labels.append(index)
                start += signal_size
                end += signal_size
    

    return data, labels



def data_transforms(normlize_type="-1-1"):
    transforms = {
        'train': aug.Compose([
            aug.Reshape(),
            aug.Normalize(normlize_type),
            aug.Retype()

        ]),
        'val': aug.Compose([
            aug.Reshape(),
            aug.Normalize(normlize_type),
            aug.Retype()
        ])
    }
    return transforms


class dataset(object):
    
    def __init__(self, data_dir, dataset, faults, signal_size, normlizetype, condition=2,
                 balance_data=False, test_size=0.2):

        self.balance_data = balance_data
        self.test_size = test_size
        self.num_classes = len(faults)
        self.train_data, self.train_labels = get_files(root=data_dir, dataset=dataset, faults=faults, signal_size=signal_size, condition=condition)
        self.transform = data_transforms(normlizetype)

    def data_preprare(self, source_label=None, random_state=1):
        
   
        data_pd = pd.DataFrame({"data": self.train_data, "labels": self.train_labels})
        data_pd = data_utils.balance_data(data_pd,random_state=42) if self.balance_data else data_pd
        

        train_dataset = data_utils.dataset(list_data=data_pd, source_label=source_label, transform=self.transform['train'])
        
        source_counts = data_pd['labels'].value_counts()
            
        source_counts = source_counts.sort_index()
    
        # Convert source_counts into a list of lists (each sublist contains label and count)
        source_counts_list = [[label, count] for label, count in source_counts.items()]
        source_counts_list.append(['Dataset Size', len(train_dataset)])  # Append dataset size at the end

        print("Source Data Distribution:")
        print(tabulate(source_counts_list, headers=['Label', 'Count'], tablefmt='psql'))

        return train_dataset
        
    
            
    