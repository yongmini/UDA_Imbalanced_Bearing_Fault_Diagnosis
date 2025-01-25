import torch
import numpy as np
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
from utils import binary_accuracy,get_accuracy
        

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, input, coeff = 1.):
        ctx.coeff = coeff
        output = input * 1.0

        return output

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha = 1.0, lo = 0.0, hi = 1.,
                      max_iters = 1000., auto_step = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input):
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo)
        if self.auto_step:
            self.step()

        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1

class DomainAdversarialLoss(nn.Module):

    def __init__(self, domain_discriminator, reduction = 'mean', grl = None):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1.,
                            max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)

    def forward(self, f_s, f_t, w_s = None, w_t = None):
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        
        d_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) \
                            + binary_accuracy(d_t, d_label_t))

        if w_s is None:
            w_s = torch.ones_like(d_label_s)
        if w_t is None:
            w_t = torch.ones_like(d_label_t)
        loss = 0.5 * (self.bce(d_s, d_label_s, w_s.view_as(d_s)) + \
                                    self.bce(d_t, d_label_t, w_t.view_as(d_t)))
        return loss, d_accuracy



def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:

    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H
    
class ConditionalDomainAdversarialLoss(nn.Module):
   
    def __init__(self, domain_discriminator: nn.Module, entropy_conditioning: bool = False,
                 randomized: bool = False, num_classes: int = -1,
                 features_dim: int = -1, randomized_dim: int = 1024,
                 reduction: str = 'mean', sigmoid=True):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = GradientReverseLayer() 
        self.entropy_conditioning = entropy_conditioning
        self.sigmoid = sigmoid
        self.reduction = reduction

   
        self.map = MultiLinearMap()
        self.bce = lambda input, target, weight: F.binary_cross_entropy(input, target, weight,
                                                                        reduction=reduction) if self.entropy_conditioning \
            else F.binary_cross_entropy(input, target, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, g_s: torch.Tensor, f_s: torch.Tensor, g_t: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f = torch.cat((f_s, f_t), dim=0)
        g = torch.cat((g_s, g_t), dim=0)
        g = F.softmax(g, dim=1).detach()
        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)

        weight = 1.0 + torch.exp(-entropy(g))
        batch_size = f.size(0)
        weight = weight / torch.sum(weight) * batch_size

        if self.sigmoid:
            d_label = torch.cat((
                torch.ones((g_s.size(0), 1)).to(g_s.device),
                torch.zeros((g_t.size(0), 1)).to(g_t.device),
            ))
            self.domain_discriminator_accuracy = binary_accuracy(d, d_label)
            if self.entropy_conditioning:
                return F.binary_cross_entropy(d, d_label, weight.view_as(d), reduction=self.reduction)
            else:
                return F.binary_cross_entropy(d, d_label, reduction=self.reduction)
        else:
            d_label = torch.cat((
                torch.ones((g_s.size(0), )).to(g_s.device),
                torch.zeros((g_t.size(0), )).to(g_t.device),
            )).long()
            self.domain_discriminator_accuracy = get_accuracy(d, d_label)
            if self.entropy_conditioning:
                raise NotImplementedError("entropy_conditioning")
            return F.cross_entropy(d, d_label, reduction=self.reduction)

class RandomizedMultiLinearMap(nn.Module):

    def __init__(self, features_dim: int, num_classes: int, output_dim: int = 1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = torch.mm(f, self.Rf.to(f.device))
        g = torch.mm(g, self.Rg.to(g.device))
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output

class MultiLinearMap(nn.Module):

    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)