'''
Paper: Ganin, Y. and Lempitsky, V., 2015, June. Unsupervised domain adaptation by backpropagation.
    In International conference on machine learning (pp. 1180-1189). PMLR.
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import wandb
import utils
import model_base
from train_utils import InitTrain
from utils import visualize_tsne_and_confusion_matrix,I_Softmax
import numpy as np     
from sklearn.metrics import confusion_matrix
import os
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import aug
import data_utils
import torch.nn as nn

class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to('cuda')
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to('cuda')

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            # print(mu.device, self.M(C,u,v).device)
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
       
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1    

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
    
class CRCL(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1):
        super(CRCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def forward(self, features, targets):
        """
        平衡的对比损失函数
        :param features: 由两组数据增强后的数据生成的feature(shape[batch_size, 2, feature_dim])
        :param targets: 特征标签(shape:[batch_size, 1])
        :return: 平衡对比损失
        """
        
        features = features.to(self.device)
        targets = targets.to(self.device)

        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets = targets.repeat(2, 1)
        batch_cls_count = torch.eye(10)[targets].sum(dim=0).squeeze()  # 统计各类别的样本数

        # mask矩阵(shape[2 * batch_size, 2 * batch_size])
        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(self.device)
        # logits_mask是对角元素为0，其他元素为1的矩阵
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(self.device),
            0
        )
        # 因为损失函数表达式的分子分母都不需要计算对自身的相似度
        # 对角元素置为0
        mask = mask * logits_mask

        features = torch.cat(torch.unbind(features, dim=1), dim=0)  # 将features切片并组合成[2 * batch_size]
        logits = features.mm(features.T)  # 计算features之间相似度
        logits = torch.div(logits, self.temperature)

        # 数值稳定性
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 负例类别平衡
        exp_logits = torch.exp(logits) * logits_mask
        # 计算公式中的Wa
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=self.device).view(1, -1).expand(
            2 * batch_size, 2 * batch_size) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)  # 对比损失的分母部分重加权

        log_prob = logits - torch.log(exp_logits_sum)  # 计算公式里的求和中的项
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # 求平均

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss
        
class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        feature_size = 2560
        self.sink = SinkhornDistance(eps=1e-3, max_iter=1000, reduction='sum') # mean으로해볼까>  별로임
        self.coral = CorrelationAlignmentLoss()
        self.model = model_base.BaseModel_add_freq(input_size=1, num_classes=args.num_classes,feature_size=feature_size,
                                      dropout=args.dropout).to(self.device)
        self._init_data()
    
    def save_model(self):
        torch.save({
            'model': self.model.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.model.load_state_dict(ckpt['model'])
        
    def train(self):
        args = self.args
        
        if args.train_mode == 'single_source':
            src = args.source_name[0]
        elif args.train_mode == 'source_combine':
            src = args.source_name
        elif args.train_mode == 'multi_source':
            raise Exception("This model cannot be trained in multi_source mode.")
        
        self.optimizer = self._get_optimizer(self.model)
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)
        
        best_acc = 0.0
        best_epoch = 0
   
        for epoch in range(1, args.max_epoch+1):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-'*5)
            
            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
   
            # Each epoch has a training and val phase
            epoch_acc = defaultdict(float)
   
            # Set model to train mode or evaluate mode
            self.model.train()
            epoch_loss = defaultdict(float)
            tradeoff = self._get_tradeoff(args.tradeoff, epoch) 
            
            num_iter = len(self.dataloaders['train'])                 
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels = utils.get_next_batch(self.dataloaders,
                						 self.iters, 'train', self.device)
                source_data, source_labels = utils.get_next_batch(self.dataloaders,
            						     self.iters, src, self.device)
                # forward
                self.optimizer.zero_grad()
                data = torch.cat((source_data, target_data), dim=0)
                y, f,es,ef = self.model(data)
                features = torch.cat([es, ef], dim=1)
                
                src_feat, tgt_feat = f.chunk(2, dim=0)
                pred, _ = y.chunk(2, dim=0)
                
                _, _, clc_loss_step = I_Softmax(2, 16, pred, source_labels,self.device).forward()
                scl_loss = CRCL(features, source_labels)
       
                loss_c = clc_loss_step
        
              #  loss_penalty, _, _ = self.sink(src_feat, tgt_feat)
                loss_penalty = self.coral(src_feat, tgt_feat)
                
              #  loss_c = F.cross_entropy(pred, source_labels)
                loss = scl_loss +  tradeoff[0] * loss_penalty
                epoch_acc['Source Data']  += utils.get_accuracy(pred, source_labels)
                
                epoch_loss['Source Classifier'] += loss_c
                epoch_loss['coral'] += loss_penalty

                # backward
                loss.backward()
                self.optimizer.step()
                
            # Print the train and val information via each epoch
            for key in epoch_acc.keys():
                avg_acc = epoch_acc[key] / num_iter
                logging.info('Train-Acc {}: {:.4f}'.format(key, avg_acc))
                wandb.log({f'Train-Acc {key}': avg_acc}, commit=False)  # Log to wandb
            for key in epoch_loss.keys():
                logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key]/num_iter))
                            
            # log the best model according to the val accuracy
            new_acc = self.test()
            
            last_acc_formatted = f"{new_acc:.3f}"
            wandb.log({"last_target_acc": float(last_acc_formatted)})
            
            
            if new_acc >= best_acc:
                best_acc = new_acc
                best_epoch = epoch
            logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch, best_acc))
            
            best_acc_formatted = f"{best_acc:.3f}"
            wandb.log({"best_target_acc": float(best_acc_formatted)})

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
        acc=self.test()
        acc_formatted = f"{acc:.3f}"
        wandb.log({"correct_target_acc": float(acc_formatted)})    
            
    def test(self):
        self.model.eval()
        acc = 0.0
        iters = iter(self.dataloaders['val'])
        num_iter = len(iters)
        with torch.no_grad():
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels, _ = next(iters)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                pred = self.model(target_data)
                acc += utils.get_accuracy(pred, target_labels)
        acc /= num_iter
        logging.info('Val-Acc Target Data: {:.4f}'.format(acc))
        return acc