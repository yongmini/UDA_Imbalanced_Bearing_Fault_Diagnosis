import torch
import numpy as np
from torch import nn
import math

# reference : https://github.com/jindongwang/transferlearning/blob/master/code/DeepDA/loss_funcs/lmmd.py

class LMMD_loss(nn.Module):
    def __init__(self, class_num=4, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, device='cuda'):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type
        self.device = device    

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label, t_label, batch_size=batch_size, class_num=self.class_num)
        weight_ss = torch.from_numpy(weight_ss).to(self.device)
        weight_tt = torch.from_numpy(weight_tt).to(self.device)
        weight_st = torch.from_numpy(weight_st).to(self.device)

        kernels = self.guassian_kernel(source, target,
                                kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).to(self.device)
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST) 
        return loss

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_label, batch_size=32, class_num=31):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.numpy()
        t_vec_label = self.convert_to_onehot(t_sca_label, class_num=self.class_num)
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')




class MMD(nn.Module):
    def __init__(self, m, n, device):
        super(MMD, self).__init__()
        self.m = m
        self.n = n
        self.device = device
    def _mix_rbf_mmd2(self, X, Y, sigma=10):
        K_XX, K_XY, K_YY = self._mix_rbf_kernel(X, Y, sigma)
        return self._mmd2(K_XX, K_XY, K_YY)

    def _mix_rbf_kernel(self, X, Y, sigma):

        X = X.to(self.device)  
        Y = Y.to(self.device)  
        XX = torch.matmul(X, X.t())
        XY = torch.matmul(X, Y.t())
        YY = torch.matmul(Y, Y.t())

        X_sqnorms = torch.diagonal(XX)
        Y_sqnorms = torch.diagonal(YY)

        r = lambda x: torch.unsqueeze(x, 0)
        c = lambda x: torch.unsqueeze(x, 1)

        K_XX, K_XY, K_YY = 0., 0., 0.

           # print(sigma,wt)
        gamma = 1 / (2 * sigma ** 2)
        K_XX =  torch.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY =  torch.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY =  torch.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))
        # print(torch.sum(torch.tensor(wts))) 1
        return K_XX, K_XY, K_YY

    def _mmd2(self, K_XX, K_XY, K_YY):

        trace_X = torch.trace(K_XX)
        trace_Y = torch.trace(K_YY)

        mmd2 = ((torch.sum(K_XX) - trace_X) / (self.m * (self.m - 1))
                + (torch.sum(K_YY) - trace_Y) / (self.n * (self.n - 1))
                - 2 * torch.sum(K_XY) / (self.m * self.n))
        return mmd2

    def _if_list(self, list):
        if len(list) == 0:
            list = torch.tensor(list)
        else:
            list = torch.vstack(list)
        return list

    def _classification_division(self, data, label):
        unique_labels = torch.unique(label)
        divided_data = {lbl.item(): [] for lbl in unique_labels}

        for i in range(data.size(0)):
            divided_data[label[i].item()].append(data[i])

        for lbl in divided_data:
            divided_data[lbl] = torch.vstack(divided_data[lbl]) if divided_data[lbl] else torch.tensor([]).to(self.device)
        
        return divided_data

    def intra_MMD(self, output1, source_label, output2, pseudo_label):
        s_data = self._classification_division(output1, source_label)
        t_data = self._classification_division(output2, pseudo_label)

        intra_MMD_loss = 0.
        class_losses = {}

        for class_label in s_data.keys():
            s_class_data = s_data[class_label]
            t_class_data = t_data.get(class_label, torch.tensor([]).to(self.device))

            if s_class_data.size(0) > 0 and t_class_data.size(0) > 0:
                #class_weights = 1  # Adjust weight as needed
                class_weights=len(pseudo_label) / t_class_data.size()[0]
                loss = class_weights * self._mix_rbf_mmd2(s_class_data, t_class_data, sigma=10)
                intra_MMD_loss += loss
                class_losses[f'class_{class_label}'] = loss

        return intra_MMD_loss, class_losses



def _update_index_matrix(batch_size, index_matrix = None):
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)

        
        for i in range(batch_size):
            for j in range(batch_size):
                index_matrix[i][j] = 1. / float(batch_size)
                index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size)

                
        for i in range(batch_size):
            for j in range(batch_size):
                index_matrix[i][j + batch_size] = -1. / float(batch_size)
                index_matrix[i + batch_size][j] = -1. / float(batch_size)
    return index_matrix


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):

    def __init__(self, kernels):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels

    def forward(self, z_s, z_t):
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size).to(z_s.device)

        # 모든 커널을 적용한 결과를 합산 --> 커널 행렬
        kernel_matrix = sum([kernel(features) for kernel in self.kernels])
        
        loss = (kernel_matrix * self.index_matrix).sum() / float(batch_size - 1)
  
        return loss


class GaussianKernel(nn.Module):

    def __init__(self, alpha = 1.):
        super(GaussianKernel, self).__init__()

        self.alpha = alpha

    def forward(self, X):
        # sample 간의 거리 matrix --> (batch_size*2, batch_size*2)
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)
        self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())
        return torch.exp(-l2_distance_square / (2 * self.sigma_square))

#https://github.com/QinYi-team/MMSD/blob/main/MMSD.ipynb
class MMSD(nn.Module):
    def __init__(self):
        super(MMSD, self).__init__()

    def _mix_rbf_mmsd(self, X, Y, sigmas=(1,), wts=None, biased=True):
        K_XX, K_XY, K_YY, d = self._mix_rbf_kernel(X, Y, sigmas, wts)
        return self._mmsd(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)

    def _mix_rbf_kernel(self, X, Y, sigmas, wts=None):
        if wts is None:
            wts = [1] * len(sigmas)
        XX = torch.matmul(X, X.t())
        XY = torch.matmul(X, Y.t())
        YY = torch.matmul(Y, Y.t())

        X_sqnorms = torch.diagonal(XX, dim1=-2, dim2=-1)
        Y_sqnorms = torch.diagonal(YY, dim1=-2, dim2=-1)

        r = lambda x: torch.unsqueeze(x, 0)
        c = lambda x: torch.unsqueeze(x, 1)

        K_XX, K_XY, K_YY = 0., 0., 0.
        for sigma, wt in zip(sigmas, wts):
            gamma = 1 / (2 * sigma ** 2)
            K_XX += wt * torch.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
            K_XY += wt * torch.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
            K_YY += wt * torch.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))
            return K_XX, K_XY, K_YY, torch.sum(torch.tensor(wts))

    def _mmsd(self, K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
        m = torch.tensor(K_XX.size(0), dtype=torch.float32)
        n = torch.tensor(K_YY.size(0), dtype=torch.float32)
        C_K_XX = torch.pow(K_XX, 2)
        C_K_YY = torch.pow(K_YY, 2)
        C_K_XY = torch.pow(K_XY, 2)
        if biased:
            mmsd = (torch.sum(C_K_XX) / (m * m) + torch.sum(C_K_YY) / (n * n)
            - 2 * torch.sum(C_K_XY) / (m * n))
        else:
            if const_diagonal is not False:
                trace_X = m * const_diagonal
                trace_Y = n * const_diagonal
            else:
                trace_X = torch.trace(C_K_XX)
                trace_Y = torch.trace(C_K_YY)

            mmsd = ((torch.sum(C_K_XX) - trace_X) / ((m - 1) * m)
                    + (torch.sum(C_K_YY) - trace_Y) / ((n - 1) * n)
                    - 2 * torch.sum(C_K_XY) / (m * n))
        return mmsd

    def forward(self, X1, X2, bandwidths=[3]):
        kernel_loss = self._mix_rbf_mmsd(X1, X2, sigmas=bandwidths)
        return kernel_loss
    
#https://github.com/liguge/Variance-discrepancy-representation-pytorch/blob/main/MVD.py
class VDR(nn.Module):
    def __init__(self):
        super(VDR, self).__init__()

    def mix_student_vdr2(self, X, Y, sigmas=(1,), wts=None, biased=False):
        K_XX, K_XY, K_YY, d = self._mix_student_kernel(X, Y, sigmas, wts)
        return self._vdr2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)

    def _mix_student_kernel(self, X, Y, sigmas, wts=None):
        if wts is None:
            wts = [1] * len(sigmas)
        XX = torch.matmul(X, X.t())
        XY = torch.matmul(X, Y.t())
        YY = torch.matmul(Y, Y.t())

        X_sqnorms = torch.diagonal(XX, dim1=-2, dim2=-1)
        Y_sqnorms = torch.diagonal(YY, dim1=-2, dim2=-1)

        r = lambda x: torch.unsqueeze(x, 0)
        c = lambda x: torch.unsqueeze(x, 1)

        K_XX, K_XY, K_YY = 0., 0., 0.
        for sigma, wt in zip(sigmas, wts):
            gamma = math.gamma((sigma+1)/2) / (math.gamma(sigma/2)*(sigma**0.5))
            K_XX += wt * gamma * torch.pow((1. + (-2 * XX + c(X_sqnorms) + r(X_sqnorms)) / 2.), -(sigma + 1.) / 2)
            K_XY += wt * gamma * torch.pow((1. + (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)) / 2.), -(sigma + 1.) / 2)
            K_YY += wt * gamma * torch.pow((1. + (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)) / 2.), -(sigma + 1.) / 2)
            return K_XX, K_XY, K_YY, torch.sum(torch.tensor(wts))

    def _vdr2(self, K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
        m = K_XX.size(0)
        n = K_YY.size(0)
        CM_m = torch.eye(m, m).cuda() - (1 / m) * torch.ones((m, m)).cuda()
        CM_n = torch.eye(n, n).cuda() - (1 / n) * torch.ones((n, n)).cuda()
        C_K_XX = torch.pow(torch.matmul(torch.matmul(CM_m, K_XX), CM_m.t()), 2)
        C_K_YY = torch.pow(torch.matmul(torch.matmul(CM_n, K_YY), CM_n.t()), 2)
        C_K_XY = torch.pow(torch.matmul(torch.matmul(CM_m, K_XY), CM_n.t()), 2)

        if biased:
            vdr2 = (torch.sum(C_K_XX) / (m * m)
                    + torch.sum(C_K_YY) / (n * n)
                    - 2 * torch.sum(C_K_XY) / (m * n))
        else:
            if const_diagonal is not False:
                trace_X = m * const_diagonal
                trace_Y = n * const_diagonal
            else:
                trace_X = torch.trace(C_K_XX)
                trace_Y = torch.trace(C_K_YY)

            vdr2 = ((torch.sum(C_K_XX) - trace_X) / ((m - 1) * m)
                    + (torch.sum(C_K_YY) - trace_Y) / ((n - 1) * n)
                    - 2 * torch.sum(C_K_XY) / (m * n))
        return vdr2

    def forward(self, X1, X2, bandwidths=[0.6]):
        kernel_loss = self.mix_student_vdr2(X1, X2, sigmas=bandwidths)
        return kernel_loss
    
    
class CorrelationAlignmentLoss(nn.Module):

    def __init__(self):
        super(CorrelationAlignmentLoss, self).__init__()

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        mean_s = f_s.mean(0, keepdim=True)
        mean_t = f_t.mean(0, keepdim=True)
        cent_s = f_s - mean_s
        cent_t = f_t - mean_t
        cov_s = torch.mm(cent_s.t(), cent_s) / (len(f_s) - 1)
        cov_t = torch.mm(cent_t.t(), cent_t) / (len(f_t) - 1)

        mean_diff = (mean_s - mean_t).pow(2).mean()
        cov_diff = (cov_s - cov_t).pow(2).mean()

        return mean_diff + cov_diff
    
#https://github.com/liguge/Deep-discriminative-transfer-learning-network-for-cross-machine-fault-diagnosis/tree/main
class DDM(nn.Module):
    def __init__(self, m, n, device):
        super(DDM, self).__init__()
        self.m = m
        self.n = n
        self.device = device
    def _mix_rbf_mmd2(self, X, Y, sigmas=(10,), wts=None, biased=True):
        K_XX, K_XY, K_YY, d = self._mix_rbf_kernel(X, Y, sigmas, wts)
        return self._mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)

    def _mix_rbf_kernel(self, X, Y, sigmas, wts=None):
        if wts is None:
            wts = [1] * len(sigmas)
        X = X.to(self.device)  
        Y = Y.to(self.device)  
        XX = torch.matmul(X, X.t())
        XY = torch.matmul(X, Y.t())
        YY = torch.matmul(Y, Y.t())

        X_sqnorms = torch.diagonal(XX)
        Y_sqnorms = torch.diagonal(YY)

        r = lambda x: torch.unsqueeze(x, 0)
        c = lambda x: torch.unsqueeze(x, 1)

        K_XX, K_XY, K_YY = 0., 0., 0.
        for sigma, wt in zip(sigmas, wts):
            gamma = 1 / (2 * sigma ** 2)
            K_XX += wt * torch.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
            K_XY += wt * torch.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
            K_YY += wt * torch.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))
            return K_XX, K_XY, K_YY, torch.sum(torch.tensor(wts))

    def _mmd2(self, K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
        if biased:
            mmd2 = torch.sum(K_XX) / (self.m * self.m) + torch.sum(K_YY) / (self.n * self.n)
            - 2 * torch.sum(K_XY) / (self.m * self.n)
        else:
            if const_diagonal is not False:
                trace_X = self.m * const_diagonal
                trace_Y = self.n * const_diagonal
            else:
                trace_X = torch.trace(K_XX)
                trace_Y = torch.trace(K_YY)

            mmd2 = ((torch.sum(K_XX) - trace_X) / (self.m * (self.m - 1))
                    + (torch.sum(K_YY) - trace_Y) / (self.n * (self.n - 1))
                    - 2 * torch.sum(K_XY) / (self.m * self.n))
        return mmd2

    def _if_list(self, list):
        if len(list) == 0:
            list = torch.tensor(list)
        else:
            list = torch.vstack(list)
        return list

    def _classification_division(self, data, label):
        unique_labels = torch.unique(label)
        divided_data = {lbl.item(): [] for lbl in unique_labels}

        for i in range(data.size(0)):
            divided_data[label[i].item()].append(data[i])

        for lbl in divided_data:
            divided_data[lbl] = torch.vstack(divided_data[lbl]) if divided_data[lbl] else torch.tensor([]).to(self.device)
        
        return divided_data

    def MDA(self, source, target, bandwidths=[10]):
        kernel_loss = self._mix_rbf_mmd2(source, target, sigmas=bandwidths) * 100.
        eps = 1e-5
        d = source.size()[1]
        ns, nt = source.size()[0], target.size()[0]

        # source covariance
        tmp_s = torch.matmul(torch.ones(size=(1, ns)).to(self.device), source)
        cs = (torch.matmul(torch.t(source), source) - torch.matmul(torch.t(tmp_s), tmp_s) / (ns + eps)) / (ns - 1 + eps) * (
                     ns / (self.m))

        # target covariance
        tmp_t = torch.matmul(torch.ones(size=(1, nt)).to(self.device), target)
        ct = (torch.matmul(torch.t(target), target) - torch.matmul(torch.t(tmp_t), tmp_t) / (nt + eps)) / (
                    nt - 1 + eps)* (
                     nt / (self.n))
        # frobenius norm
        # loss = torch.sqrt(torch.sum(torch.pow((cs - ct), 2)))
        loss = torch.norm((cs - ct))
        loss = loss / (4 * d * d) * 10.

        return loss + kernel_loss


    def CDA(self, output1, source_label, output2, pseudo_label):
        s_data = self._classification_division(output1, source_label)
        t_data = self._classification_division(output2, pseudo_label)

        intra_MMD_loss = 0.
    

        for class_label in s_data.keys():
            s_class_data = s_data[class_label]
            t_class_data = t_data.get(class_label, torch.tensor([]).to(self.device))

            if s_class_data.size(0) > 0 and t_class_data.size(0) > 0:

                loss = self.MDA(s_class_data, t_class_data)
                intra_MMD_loss += loss
       

        return intra_MMD_loss
    


