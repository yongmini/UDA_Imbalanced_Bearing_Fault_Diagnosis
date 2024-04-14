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
                start_train += signal_size // 2  # 50% overlap
                end_train += signal_size // 2
            
            # Testing data with no overlap
            start_test, end_test = half, half + signal_size
            while end_test <= signal.shape[0]:
                test_data.append(signal[start_test:end_test])
                test_labels.append(index)
                start_test += signal_size
                end_test += signal_size
    
    print("Training data length:", len(train_data))
    print("Testing data length:", len(test_data))
    return train_data, train_labels, test_data, test_labels


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
        self.train_data, self.train_labels,self.test_data,self.test_labels = get_files(root=data_dir, dataset=dataset, faults=faults, signal_size=signal_size, condition=condition)
        self.transform = data_transforms(normlizetype)

    def data_preprare(self, source_label=None, is_src=False, random_state=1, imbalance_ratio=None):
        
        self.combined_data = self.train_data + self.test_data
        self.combined_labels = self.train_labels + self.test_labels
        data_pd = pd.DataFrame({"data": self.combined_data, "labels": self.combined_labels})
        data_pd = data_utils.balance_data(data_pd,random_state=random_state) if self.balance_data else data_pd
        
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
            
    
            train_pd = pd.DataFrame({"data": self.train_data, "labels": self.train_labels})
            train_pd = data_utils.balance_data(train_pd,random_state=random_state)
            val_pd = pd.DataFrame({"data": self.test_data, "labels": self.test_labels})
            val_pd = data_utils.balance_data(val_pd,random_state=random_state)
            
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
            #######
                # val_data = []
                # val_labels = []
                # for label, ratio in imbalance_ratio.items():
                #     num_samples = int(len(val_pd[val_pd["labels"] == label]) * ratio)
                #     val_data += val_pd[val_pd["labels"] == label]["data"].tolist()[:num_samples]
                #     val_labels += val_pd[val_pd["labels"] == label]["labels"].tolist()[:num_samples]
                    
                # val_pd = pd.DataFrame({"data": train_data, "labels": train_labels})           
            
            
            # DataFrame 형태의 `train_pd`를 CSV 파일로 저장
            
            #train_pd.to_csv("train_csv_path.csv", index=False)
            

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
