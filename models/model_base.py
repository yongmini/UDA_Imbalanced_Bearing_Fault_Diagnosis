
import torch.nn as nn
from models.resnet1d import FeatureExtractor

class ClassifierMLP(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 dropout,
                 last='tanh'):
        super(ClassifierMLP, self).__init__()
        
        self.last = last
        self.net = nn.Sequential(  
                   nn.Linear(input_size, int(input_size/4)),
                   nn.ReLU(),
                   nn.Dropout(p=dropout),  
                   nn.Linear(int(input_size/4), output_size))
              
        
        if last == 'logsm':
            self.last_layer = nn.LogSoftmax(dim=-1)
        elif last == 'sm':
            self.last_layer = nn.Softmax(dim=-1)
        elif last == 'tanh':
            self.last_layer = nn.Tanh()
        elif last == 'sigmoid':
            self.last_layer = nn.Sigmoid()
        elif last == 'relu':
            self.last_layer = nn.ReLU()

    def forward(self, input):
        y = self.net(input)
        if self.last != None:
            y = self.last_layer(y)
        
        return y
    
class BaseModel(nn.Module):
    
    def __init__(self,
                 input_size,
                 num_classes,
                 dropout):
        super(BaseModel, self).__init__()
        
        self.G = FeatureExtractor(in_channel=input_size)
        
        self.C = ClassifierMLP(512, num_classes, dropout, last=None)
        
    def forward(self, input):
        f = self.G(input)
        predictions = self.C(f)
        
        if self.training:
            return predictions, f
        else:
            return predictions