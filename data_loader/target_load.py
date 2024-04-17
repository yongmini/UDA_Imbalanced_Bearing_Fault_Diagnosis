import os
import importlib
import pandas as pd
from scipy.io import loadmat
from tabulate import tabulate
import aug
import data_utils
import load_methods




def get_files(root, dataset, faults, signal_size, condition=3):
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    data_load = getattr(load_methods, dataset)
    
    for index, name in enumerate(faults):
        data_dir = os.path.join(root, f'condition_{condition}', name)
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            signal = data_load(item_path)
            
            half = signal.shape[0] // 2  # Find the midpoint of the signal
            
            # Training data with 50% overlap
            start_train, end_train = 0, signal_size
            while end_train <= half:
                train_data.append(signal[start_train:end_train])
                train_labels.append(index)
                start_train += signal_size  #// 2  # 50% overlap
                end_train += signal_size# // 2
            
            # Testing data with no overlap
            start_test, end_test = half, half + signal_size
            while end_test <= signal.shape[0]:
                test_data.append(signal[start_test:end_test])
                test_labels.append(index)
                start_test += signal_size
                end_test += signal_size
    

    return train_data, train_labels, test_data, test_labels


def data_transforms(normlize_type="-1-1"):
    transforms = {
        'train': aug.Compose([
            aug.Reshape(),
            aug.Normalize(normlize_type),
           # aug.RandomShuffleSegments(),
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
                 balance_data=True, test_size=0.2):

        self.balance_data = balance_data
        self.test_size = test_size
        self.num_classes = len(faults)
        self.target_train_data, self.target_train_labels,self.target_test_data,self.target_test_labels = get_files(root=data_dir, dataset=dataset, faults=faults, signal_size=signal_size, condition=condition)
        self.transform = data_transforms(normlizetype)

    def data_preprare(self, source_label=None, random_state=1, imbalance_ratio=None):
        

        
        train_pd = pd.DataFrame({"data": self.target_train_data, "labels": self.target_train_labels})
        train_pd = data_utils.balance_data(train_pd,random_state=random_state)
        val_pd = pd.DataFrame({"data": self.target_test_data, "labels": self.target_test_labels})
        val_pd = data_utils.balance_data(val_pd,random_state=42)
        
        if imbalance_ratio is not None:
            train_data = []
            train_labels = []
            for label, ratio in imbalance_ratio.items():
                # Calculate the number of samples to select for the current label
                num_samples = int(len(train_pd[train_pd["labels"] == label]) * ratio)
                
                # Randomly sample the required number of entries
                sampled_data = train_pd[train_pd["labels"] == label].sample(n=num_samples, random_state=random_state)
                
                # Extract data and labels from the sampled dataframe
                train_data += sampled_data["data"].tolist()
                train_labels += sampled_data["labels"].tolist()
            
            # Create a new DataFrame with the sampled data and labels
            train_pd = pd.DataFrame({"data": train_data, "labels": train_labels})

 
        train_dataset = data_utils.dataset(list_data=train_pd, source_label=source_label, transform=self.transform['train'])
        val_dataset = data_utils.dataset(list_data=val_pd, source_label=source_label, transform=self.transform['val'])

        # Prepare to print data distributions for training and validation sets
        target_counts = pd.concat([train_pd['labels'].value_counts(), val_pd['labels'].value_counts()], axis=1, keys=['train', 'val'])
        target_counts = target_counts.sort_index()
        target_counts.sort_index(inplace=True)
        target_counts.loc['Dataset Size'] = [len(train_dataset), len(val_dataset)]
        print("Target Data Distribution:")
        print(tabulate(target_counts, headers='keys', tablefmt='psql'))

        return train_dataset, val_dataset
