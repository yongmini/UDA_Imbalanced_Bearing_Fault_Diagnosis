import torch
import numpy as np
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

class I_Softmax(nn.Module):
    def __init__(self, m, n,cls_num_list, source_output1, source_label, device):
        super().__init__()
        
        self.device = device
        self.m = torch.tensor([m]).to(self.device)
        self.n = torch.tensor([n]).to(self.device)
        self.source_output1 = source_output1
        self.source_label = source_label
        self.num_classes = len(cls_num_list)
        self.class_data = {i: [] for i in range(self.num_classes)}
        self.class_labels = {i: [] for i in range(self.num_classes)}
        self.data_set = []
        self.label_set = []


    def _combine(self):
        for i in range(self.source_label.size()[0]):
            label = self.source_label[i].item()
            self.class_data[label].append(self.source_output1[i])
            self.class_labels[label].append(self.source_label[i])

        for label in range(self.num_classes):
            if len(self.class_data[label]) > 0:
                processed_data = self._class_angle(self.class_data[label], self.class_labels[label])
                if processed_data is not None:
                    self.data_set.append(processed_data)
                    self.label_set.append(torch.tensor(self.class_labels[label]).unsqueeze(1))
        if len(self.data_set) > 0:
            data = torch.vstack(self.data_set)
            label = torch.vstack(self.label_set)
            return data.to(self.device), label.squeeze().to(self.device)
        return None, None

    def _class_angle(self, class_data, class_labels):
        if len(class_labels) == 0:
            return None
        
        index = class_labels[0]
        new_tensor = None
        
        for i, c in enumerate(class_data):
            if len(c.shape) == 1:
                c = c.unsqueeze(0)
            
            part1 = c[:, :index]
            part2 = c[:, index + 1:]
            
            if c[:, index] > 0:
                val = c[:, index] / (self.m + 1e-5) - self.n
            else:
                val = c[:, index] * (self.m + 1e-5) - self.n
            
            val = val.view(-1, 1)
            tensor = torch.cat((part1, val, part2), dim=1)
            
            if new_tensor is None:
                new_tensor = tensor
            else:
                new_tensor = torch.vstack([new_tensor, tensor])
                
        return new_tensor

    def forward(self):
        data, label = self._combine()
        
  

        loss = torch.nn.functional.cross_entropy(data, label, reduction='mean',label_smoothing=0.1)

        return data, label, loss


    
# reference : https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
class LDAM_Loss(nn.Module):
    
    def __init__(self, cls_num_list,weight, device, max_m=0.5, s=30):
        super(LDAM_Loss, self).__init__()
        self.max_m = max_m
        assert s > 0
        self.s = s
        
        self.cls_num_list = cls_num_list
        self.weight = weight
        self.device = device

    def forward(self, x, target):
        device = target.device
        weight = self.weight.to(device)
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)

        m_list = 1.0 / np.sqrt(np.sqrt(self.cls_num_list))
        m_list = m_list * (self.max_m / np.max(m_list))
        m_list = torch.tensor(m_list, dtype=torch.float32, device=self.device)
        self.m_list = m_list

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
      
        return F.cross_entropy(self.s*output, target, weight=weight, label_smoothing=0.1)


# reference :  https://github.com/agaldran/cost_sensitive_loss_classification/blob/master/utils/losses.py

class CostSensitiveLoss(nn.Module):
    def __init__(self, n_classes, tr=1, reduction='mean'):
        super(CostSensitiveLoss, self).__init__()
        
        self.n_classes = n_classes
        self.reduction = reduction
        self.tr = tr
        self.normalization = nn.Softmax(dim=1)


    def calculate_weights(self, target):

        class_counts = np.bincount(target.cpu().numpy(), minlength=self.n_classes)
    
        nt = len(target)

        weights = self.n_classes * (nt / class_counts) 
        weights[np.isinf(weights)] = 1

        return torch.from_numpy(weights).float()
    
    def cost_sensitive_loss(self, input, target, weights):
        if input.size(0) != target.size(0):
            raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                            .format(input.size(0), target.size(0)))
        device = input.device
        weights = weights.to(device)
        loss = torch.nn.functional.cross_entropy(input, target, reduction='none')#,label_smoothing=0.1)
        if self.tr == 0:
            weighted_loss =  loss
        else:   
            weighted_loss = weights[target] * loss
        return weighted_loss
    
    def forward(self, logits, target):
        weights = self.calculate_weights(target)

        loss = self.cost_sensitive_loss(logits, target, weights)
        
        if self.reduction == 'none':    
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError('`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

