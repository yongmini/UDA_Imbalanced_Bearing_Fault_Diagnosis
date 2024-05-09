import torch
import numpy as np
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import seaborn as sns


def get_accuracy(preds, targets):
        assert preds.shape[0] == targets.shape[0]
        correct = torch.eq(preds.argmax(dim=1), targets).float().sum().item()
        accuracy = correct/preds.shape[0]
        return accuracy


def binary_accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


def gmean(iterable):
    a = np.array(iterable)
    return a.prod() ** (1. / len(a))


def freeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = True


def get_next_batch(loaders, iters, src, device, return_idx=False):
    inputs, labels = None, None
    if type(src) == list:
        for key in src:
            try:
                inputs, labels, src_idx = next(iters[key])
                break
            except StopIteration:
                continue
        if inputs == None:
            for key in src:
                iters[key] = iter(loaders[key])
            inputs, labels, src_idx = next(iters[src[0]])
    else:
        try:
            inputs, labels, src_idx = next(iters[src])
        except StopIteration:
            iters[src] = iter(loaders[src])
            inputs, labels, src_idx = next(iters[src])
    
    if return_idx:
        return inputs.to(device), labels.to(device), src_idx.to(device)
    else:
        return inputs.to(device), labels.to(device)
    
def get_next_batch_balanced(loaders,device):
    inputs, labels = None, None
    inputs, labels = iter(loaders)
    return inputs.to(device), labels.to(device)

def get_concat_dataset_next_batch(loaders, iters, src, device, return_idx=False):
    inputs, labels = None, None
    try:
        inputs, labels, src_idx = next(iters[src])
    except StopIteration:
        iters[src] = iter(loaders[src])
        inputs, labels, src_idx = next(iters[src])
    
    if return_idx:
        return inputs.to(device), labels.to(device), src_idx.to(device)
    else:
        return inputs.to(device), labels.to(device)


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


def _update_index_matrix(batch_size, index_matrix = None, linear = True):
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            '''
            # Seems that this part is wrong.
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
            '''
            # The following is the revised version.
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j] = 1. / float(batch_size)
                    index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size)
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)

    return index_matrix


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):

    def __init__(self, kernels, linear = True):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s, z_t):
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)

        # Add up the matrix of each kernel
        kernel_matrix = sum([kernel(features) for kernel in self.kernels])
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        if self.linear:
            loss = (kernel_matrix * self.index_matrix).sum() / float(batch_size - 1)
        else:
            loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss


class GaussianKernel(nn.Module):

    def __init__(self, sigma = None, track_running_stats = True, alpha = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X):
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


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


def visualize_tsne_and_confusion_matrix(features, all_labels, cm, save_dir,file_name):
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    transformed_features = tsne.fit_transform(features)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Visualize t-SNE
    for label in np.unique(all_labels):
        ax1.scatter(transformed_features[all_labels == label, 0], transformed_features[all_labels == label, 1], label=label)
    ax1.set_title('t-SNE Visualization')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.legend()

    # Visualize confusion matrix
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=list(range(len(np.unique(all_labels)))), yticklabels=list(range(len(np.unique(all_labels)))), ax=ax2)
    ax2.set_title('Confusion Matrix')
    ax2.set_xlabel('Predicted Labels')
    ax2.set_ylabel('True Labels')

    # Save the combined plot
    save_path = os.path.join(save_dir,file_name)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
class I_Softmax(nn.Module):
    def __init__(self, m, n, source_output1, source_label, device):
        super().__init__()
        self.device = device
        self.m = torch.tensor([m]).to(self.device)
        self.n = torch.tensor([n]).to(self.device)
        self.source_output1 = source_output1
        self.source_label = source_label
        self.class_data = {}
        self.class_labels = {}
        self.data_set = []
        self.label_set = []

    def _combine(self):
        for i in range(self.source_label.size()[0]):
            label = self.source_label[i].item()
            if label not in self.class_data:
                self.class_data[label] = []
                self.class_labels[label] = []
            self.class_data[label].append(self.source_output1[i])
            self.class_labels[label].append(label)

        for label, data in self.class_data.items():
            class_data_tensor = torch.stack(data)
            class_label_tensor = torch.tensor(self.class_labels[label]).unsqueeze(1)

            self.data_set.append(self._class_angle(class_data_tensor, class_label_tensor))
            self.label_set.append(class_label_tensor)

        data = torch.vstack(self.data_set)
        label = torch.vstack(self.label_set)
        return data.to(self.device), label.squeeze().to(self.device)
    

    def _class_angle(self, a, la):

        if len(la) == 0:
            return a
        else:
            index = la[0]
        for i in range(len(a)):
            c = a[i]
            part1 = c[:index]
            part2 = c[index + 1:]
            if c[index] > 0:
                val = c[index] / (self.m + 1e-5) - self.n
            elif c[index] <= 0:
                val = c[index] * (self.m + 1e-5) - self.n
            if i == 0:
                new_tensor = torch.concat((part1, val, part2))
            else:
                tensor = torch.concat((part1, val, part2), dim=0)
                new_tensor = torch.vstack([new_tensor, tensor])

        return new_tensor

    def forward(self):
        data, label = self._combine()
        loss = F.nll_loss(F.log_softmax(data, dim=-1), label)
        return data, label, loss
    