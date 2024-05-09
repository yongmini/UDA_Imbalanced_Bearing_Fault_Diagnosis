import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet1d import FeatureExtractor
import torch.fft as fft
import math
from torch.autograd import Variable
from scipy.special import binom

class LSoftmaxLinear(nn.Module):

    def __init__(self, input_features, output_features, margin, device):
        super().__init__()
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99

        self.device = device  # gpu or cpu

        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).to(device)  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).to(device)  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).to(device)  # n
        self.signs = torch.ones(margin // 2 + 1).to(device)  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta**2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            x, w = input, self.weight
            beta = max(self.beta, self.beta_min)
            logit = x.mm(w)
            indexes = range(logit.size(0))
            logit_target = logit[indexes, target]

            # cos(theta) = w * x / ||w||*||x||
            w_target_norm = w[:, target].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            # equation 7
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

            # find k in equation 6
            k = self.find_k(cos_theta_target)

            # f_y_i
            logit_target_updated = (w_target_norm *
                                    x_norm *
                                    (((-1) ** k * cos_m_theta_target) - 2 * k))
            logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta)

            logit[indexes, target] = logit_target_updated_beta
            self.beta *= self.scale
            return logit
        else:
            assert target is None
            return input.mm(self.weight)
        
        
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
    
    
class LsoftmaxClassifierMLP(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 dropout,
                 device):
        super(LsoftmaxClassifierMLP, self).__init__()
        self.margin = 4
        self.device= device
        self.fc = nn.Sequential(  
                   nn.Linear(input_size, int(input_size/4)),
                   nn.ReLU(),
                   nn.Dropout(p=dropout)
                   )
                    
                 
        self.lsoftmax_linear = LSoftmaxLinear(
                input_features=(int(input_size/4)), output_features=output_size, margin=self.margin,device=self.device)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lsoftmax_linear.reset_parameters()
        
    def forward(self, input, target=None):
        x = self.fc(input)
        logit = self.lsoftmax_linear(input=x, target=target)

        
        return logit


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

class BaseModel_lsoft(nn.Module):
    
    def __init__(self,
                 input_size,
                 num_classes,
                 dropout):
        super(BaseModel_lsoft, self).__init__()
        
        self.G = FeatureExtractor(in_channel=input_size)
        
        self.C = LsoftmaxClassifierMLP(512, num_classes, dropout,device='cuda')
        
    def forward(self, input, target=None):
        f = self.G(input)
        predictions = self.C(f,target)
        
        if self.training:
            return predictions, f
        else:
            return predictions

class BaseModel_add_freq_lsoft(nn.Module):
    
    def __init__(self,
                 input_size,
                 feature_size,
                 num_classes,
                 dropout):
        super(BaseModel_add_freq_lsoft, self).__init__()
        
        self.G = FeatureExtractor(in_channel=input_size)
        
        self.S = SpectralConv1d(1, 1)
        #self.C = ClassifierMLP(feature_size, num_classes, dropout, last=None)
        self.C = LsoftmaxClassifierMLP(2560, num_classes, dropout,device='cuda')
        
    def forward(self, input, target=None):
        es = self.G(input)
        ef =  self.S(input) 

        f = torch.concat([ef,es],-1)
    
        f= F.normalize(f) #필요해
        predictions = self.C(f, target)
        
        
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
        
        


class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=1, padding=7),  #(number of kernels, kernel size, padding) - (pooling size, pooling stride)
            nn.MaxPool1d(2, stride=2), # 1024
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 7, stride=1, padding=3), 
            nn.MaxPool1d(2, stride=2), # 512
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=1, padding=2), 
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2, stride=2), # 256
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 256, 512)
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.layers(x)
    
class Shared_feature_decoder(nn.Module):
    def __init__(self):
        super(Shared_feature_decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 64 * 256),
            nn.ReLU(),
            nn.Unflatten(1, (64, 256)), 
            nn.ConvTranspose1d(64, 64, 5, stride=2, padding=2, output_padding=1),  #  #(number of kernels, kernel size, padding) - (pooling size, pooling stride)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 15, stride=2, padding=7, output_padding=1),  
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.layers(x)

 class BaseModel_recon(nn.Module):
    
    def __init__(self,
                 input_size,
                 num_classes,
                 dropout):
        super(BaseModel_recon, self).__init__()
        
        self.G = FeatureEncoder(in_channel=input_size)
        self.D = 
        
        self.C = ClassifierMLP(512, num_classes, dropout, last=None)
        
    def forward(self, input):
        f = self.G(input)
        predictions = self.C(f)
        
        if self.training:
            return predictions, f
        else:
            return predictions       
        
        
# class SpectralConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(SpectralConv1d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
    

#         self.weight = nn.Parameter(torch.empty((1025, in_channels, out_channels), dtype=torch.cfloat))
#         self.bias = nn.Parameter(torch.empty((1025, out_channels), dtype=torch.cfloat))
#         self.reset_parameters()
        
#     def forward(self, input):
#         # Forward Fourier Transform
    
#         # input - b t d
#         b, t, _ = input.shape
#         input_fft = fft.rfft(input, dim=1)
#         output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
#         output_fft = self._forward(input_fft)
#         r = output_fft.abs()
#         p = output_fft.angle() 
#         result=  torch.concat([r,p],-1)
        
#         #result = fft.irfft(output_fft, n=input.size(1), dim=1, norm='ortho')
#         result = F.relu(result)
       
#         result = result.flatten(start_dim=1)
#         return result  

    
#     def _forward(self, input):
#         output = torch.einsum('bti,tio->bto', input, self.weight)
#         return output  + self.bias

#     def reset_parameters(self) -> None:
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#         nn.init.uniform_(self.bias, -bound, bound)
    
    