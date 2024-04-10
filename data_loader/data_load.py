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
        data_dir = os.path.join(root, 'condition_%d' % condition, name)
    #    print(data_dir)
        for item in os.listdir(data_dir):
            
            item_path = os.path.join(data_dir, item)
      #      print(item_path)
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
                 balance_data=True, test_size=0.2):
        self.balance_data = balance_data
        self.test_size = test_size
        self.num_classes = len(faults)
        self.data, self.labels = get_files(root=data_dir, dataset=dataset, faults=faults, signal_size=signal_size, condition=condition)
        self.transform = data_transforms(normlizetype)

    def data_preprare(self, source_label=None, is_src=False, random_state=1, imbalance_ratio=None):
        data_pd = pd.DataFrame({"data": self.data, "labels": self.labels})
        data_pd = data_utils.balance_data(data_pd) if self.balance_data else data_pd
        if is_src:
            train_dataset = data_utils.dataset(list_data=data_pd, source_label=source_label, transform=self.transform['train'])
            
            

            source_counts = data_pd['labels'].value_counts()
              
            source_counts = source_counts.sort_index()
      
            # Convert source_counts into a list of lists (each sublist contains label and count)
            source_counts_list = [[label, count] for label, count in source_counts.items()]
            source_counts_list.append(['Dataset Size', len(train_dataset)])  # Append dataset size at the end

            print("Source Data Distribution:")
            print(tabulate(source_counts_list, headers=['Label', 'Count'], tablefmt='psql'))

            return train_dataset
        
        else:
            
    
            train_pd = pd.DataFrame({"data": self.data, "labels": self.labels})
            train_pd, val_pd = data_utils.train_test_split_(data_pd, test_size=self.test_size, num_classes=self.num_classes, random_state=random_state)
            
            if imbalance_ratio is not None:
                train_data = []
                train_labels = []
                for label, ratio in imbalance_ratio.items():
                    num_samples = int(len(train_pd[train_pd["labels"] == label]) * ratio)
                    train_data += train_pd[train_pd["labels"] == label]["data"].tolist()[:num_samples]
                    train_labels += train_pd[train_pd["labels"] == label]["labels"].tolist()[:num_samples]
                    
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
