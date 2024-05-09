import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet1d import FeatureExtractor
import torch.fft as fft
import math
from torch.autograd import Variable
from scipy.special import binom


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
    
        # Initialize weights as complex numbers
        self.freq_weights_real = nn.Parameter(torch.rand(out_channels, in_channels, 1))
        self.freq_weights_imag = nn.Parameter(torch.rand(out_channels, in_channels, 1))

    def forward(self, x):
        # Forward Fourier Transform
        x_ft = torch.fft.rfft(x, norm='ortho')
        
        # Frequency dimension
        freq_size = x_ft.shape[-1]
        
        # Expand the weights to match the frequency dimension
        freq_weights_real = self.freq_weights_real.expand(-1, -1, freq_size)
        freq_weights_imag = self.freq_weights_imag.expand(-1, -1, freq_size)
        
        # Create complex weights
        freq_weights = torch.complex(freq_weights_real, freq_weights_imag)
        
        # Complex multiplication
        out_freq = torch.einsum('bif,oif->bof', x_ft, freq_weights)
        

        result = torch.fft.irfft(out_freq, n=x.shape[-1], norm='ortho')
        result = F.relu(result) #무조건있어야네
        result = result.flatten(start_dim=1)
        return result   
    

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


class BaseModel_add_freq(nn.Module):
    
    def __init__(self,
                 input_size,
                 feature_size,
                 num_classes,
                 dropout):
        super(BaseModel_add_freq, self).__init__()
        
        self.G = FeatureExtractor(in_channel=input_size)
        
        self.S = SpectralConv1d(1, 1)
        self.C = ClassifierMLP(feature_size, num_classes, dropout, last=None)
    
    def forward(self, input):
        es = self.G(input)
        ef =  self.S(input) 

        f = torch.concat([ef,es],-1)
    
        f= F.normalize(f) #필요해
        predictions = self.C(f)
        
        
        if self.training:
            return predictions, f
        else:
            return predictions        
        
        
# class FeatureEncoder(nn.Module):
#     def __init__(self):
#         super(FeatureEncoder, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Conv1d(1, 32, 15, stride=1, padding=7),
#             nn.MaxPool1d(2, stride=2),  # Output size: 1024 -> 512
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Conv1d(32, 64, 7, stride=1, padding=3),
#             nn.MaxPool1d(2, stride=2),  # Output size: 512 -> 256
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Conv1d(64, 64, 5, stride=1, padding=2),
#             nn.BatchNorm1d(64),
#             nn.MaxPool1d(2, stride=2),  # Output size: 256 -> 128
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(64 * 128, 512)
#         )
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, x):
#         return self.layers(x)
    
# class SharedFeatureDecoder(nn.Module):
#     def __init__(self):
#         super(SharedFeatureDecoder, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(512, 64 * 256),
#             nn.ReLU(),
#             nn.Unflatten(1, (64, 256)), 
#             nn.ConvTranspose1d(64, 64, 5, stride=2, padding=2, output_padding=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.ConvTranspose1d(64, 32, 7, stride=2, padding=3, output_padding=1),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.ConvTranspose1d(32, 1, 15, stride=2, padding=7, output_padding=1)
#         )
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, x):
#         return self.layers(x)

class ClassifierMLP(nn.Module):
    def __init__(self, input_size, num_classes, dropout, last=None):
        super(ClassifierMLP, self).__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        self.last = last

    def forward(self, x):
        x = self.linear_layers(x)
        if self.last:
            x = self.last(x)
        return x

# class BaseModelRecon(nn.Module):
#     def __init__(self, input_size, num_classes, dropout):
#         super(BaseModelRecon, self).__init__()
#         self.G = FeatureEncoder()
#         self.D = SharedFeatureDecoder()
#         self.C = ClassifierMLP(512, num_classes, dropout, last=None)

#     def forward(self, input):
#         f = self.G(input)
#         r = self.D(f)
#         predictions = self.C(f)
#         if self.training:
#             return predictions, f, r
#         else:
#             return predictions
        
    
    