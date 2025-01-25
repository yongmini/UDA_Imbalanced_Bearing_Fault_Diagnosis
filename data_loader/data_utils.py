import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import aug

        
def balance_data(data_pd, random_state=10):
    
    count = data_pd['labels'].value_counts()        
    min_len = min(count)                          
    df = pd.DataFrame(columns=('data', 'labels'))
    rng = np.random.default_rng(random_state)  
    for i in count.keys():
        data_pd_tmp = data_pd[data_pd['labels'] == i].reset_index(drop=True)
        indices = rng.choice(data_pd_tmp.index, min_len, replace=False)   
        df = pd.concat([df, data_pd_tmp.loc[indices, ['data', 'labels']]], ignore_index=True)
    return df



class dataset(Dataset):
    def __init__(self, list_data, transform=None):

        self.seq_data = list_data['data'].tolist()
        self.labels = list_data['labels'].tolist()


        if transform is None:
            self.transforms = aug.Compose([
                aug.Reshape(),
                aug.Retype()
            ])
        else:
            self.transforms = transform

    def __len__(self):
     
        return len(self.seq_data)

    def __getitem__(self, item):
   
        seq = self.seq_data[item]
        label = self.labels[item]


        seq = self.transforms(seq)
        return seq, label
    

            
