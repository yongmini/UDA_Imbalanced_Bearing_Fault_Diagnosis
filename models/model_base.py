import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet1d import FeatureExtractor
import torch.fft as fft
import math

# class SpectralConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1):
#         super(SpectralConv1d, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1

#         self.scale = (1 / (in_channels * out_channels))
        
#         self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 2048 // 2 + 1, dtype=torch.cfloat))

#         self.flatten = nn.Flatten()  # Flatten layer

#     # Complex multiplication
#     def compl_mul1d(self, input, weights):
#         return torch.einsum("bix,iox->box", input, weights)

#     def forward(self, x):
#         batchsize = x.shape[0]

#         x_ft = torch.fft.rfft(x, norm='ortho')
#         out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1,  device=x.device, dtype=torch.cfloat)
#         out_ft = self.compl_mul1d(x_ft, self.weights1)

#         r = x_ft[:, :, :self.modes1].abs()
#         p = x_ft[:, :, :self.modes1].angle()
#         f = torch.concat([r, p], -1)
#         f = self.flatten(f)

#         return f

# class SpectralConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(SpectralConv1d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.scale = (1 / (in_channels * out_channels))
        
#         # 주파수에 대한 가중치를 초기화합니다.
#         self.freq_weights = nn.Parameter(torch.rand(out_channels, in_channels, 1))  # 1은 임시 차원입니다.

#     def forward(self, x):
#         # FFT를 수행하여 주파수 영역으로 변환합니다.
#         x_ft = torch.fft.rfft(x, norm='ortho')

#         # 주파수(진폭)을 계산합니다.
#         r = x_ft.abs()

#         # FFT 결과의 크기를 계산하여 가중치 차원을 조정합니다.
#         freq_size = r.shape[-1]
#         self.freq_weights = nn.Parameter(self.freq_weights.expand(-1, -1, freq_size))

#         # 주파수에 가중치를 적용합니다.
#         out_freq = torch.einsum("bix,iox->box", r, self.freq_weights)

#         # 결과를 비선형 활성화 함수를 통과시킵니다.
#         #out_freq = torch.relu(out_freq)

#         # 결과를 평탄화
#         result = out_freq.flatten(start_dim=1)

#         return result

# class SpectralConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(SpectralConv1d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
    
        
#         self.freq_weights = nn.Parameter(torch.rand(out_channels, in_channels, 1))



#     def forward(self, x):

#         x_ft = torch.fft.rfft(x, norm='ortho')
#         x_ft = x_ft.real

#         freq_size = x_ft.shape[-1]
#         freq_weights = self.freq_weights.expand(-1, -1, freq_size)

#         out_freq = torch.einsum('bif,oif->bof', x_ft, freq_weights)

#         result = out_freq.flatten(start_dim=1)

#         return result
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