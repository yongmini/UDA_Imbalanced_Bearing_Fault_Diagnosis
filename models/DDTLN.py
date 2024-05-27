'''
Paper: Qian, Q., Qin, Y., Luo, J., Wang, Y., and Wu, F. (2023). 
Deep discriminative transfer learning network for cross-machine fault diagnosis. 
Mechanical Systems and Signal Processing, 186, 109884.
Reference code: https://github.com/QinYi-team/Code/blob/master/DDTLN/DDTLN%20code.ipynb
'''
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import wandb
import utils
import model_base
from train_utils import InitTrain



class MMD(nn.Module):
    def __init__(self, m, n, device):
        super(MMD, self).__init__()
        self.m = m
        self.n = n
        self.device = device
    def _mix_rbf_mmd2(self, X, Y, sigmas=(10,), wts=None, biased=True):
        K_XX, K_XY, K_YY, d = self._mix_rbf_kernel(X, Y, sigmas, wts)
        return self._mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)

    def _mix_rbf_kernel(self, X, Y, sigmas, wts=None):
        if wts is None:
            wts = [1] * len(sigmas)
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
        N = data.size()[0]
        a, b, c, d = [], [], [], []
        for i in range(N):
            if label[i] == 0:
                a.append(data[i])
            elif label[i] == 1:
                b.append(data[i])
            elif label[i] == 2:
                c.append(data[i])
            elif label[i] == 3:
                d.append(data[i])

        return self._if_list(a), self._if_list(b), self._if_list(c), self._if_list(d)

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
        s_0, s_1, s_2, s_3 = self._classification_division(output1, source_label)
        t_0, t_1, t_2, t_3 = self._classification_division(output2, pseudo_label)

        CDA_loss = 0.
        if t_0.size()[0] != 0:
            CDA_loss += self.MDA(s_0, t_0)
        if t_1.size()[0] != 0:
            CDA_loss += self.MDA(s_1, t_1)
        if t_2.size()[0] != 0:
            CDA_loss += self.MDA(s_2, t_2)
        if t_3.size()[0] != 0:
            CDA_loss += self.MDA(s_3, t_3)
        return CDA_loss / 4.

#############################################################33
class I_Softmax(nn.Module):

    def __init__(self, m, n, source_output1, source_label, device):
        super().__init__()
        self.device = device  # 사용할 디바이스 (CPU 또는 GPU)
        self.m = torch.tensor([m]).to(self.device)  # 변형 과정에서 사용할 m 파라미터를 텐서로 변환하여 디바이스에 할당
        self.n = torch.tensor([n]).to(self.device) 
        self.source_output1 = source_output1
        self.source_label = source_label
        self.la, self.lb, self.lc, self.ld = [], [], [], []
        self.a, self.b, self.c, self.d = [], [], [], []
        self.data_set = []
        self.label_set = []

    def _combine(self):
        for i in range(self.source_label.size()[0]):
            if self.source_label[i] == 0:
                self.a.append(self.source_output1[i])
                self.la.append(self.source_label[i])
            elif self.source_label[i] == 1:
                self.b.append(self.source_output1[i])
                self.lb.append(self.source_label[i])
            elif self.source_label[i] == 2:
                self.c.append(self.source_output1[i])
                self.lc.append(self.source_label[i])
            elif self.source_label[i] == 3:
                self.d.append(self.source_output1[i])
                self.ld.append(self.source_label[i])

        a = self._class_angle(self.a, self.la)
        b = self._class_angle(self.b, self.lb)
        c = self._class_angle(self.c, self.lc)
        d = self._class_angle(self.d, self.ld)

        if len(a) != 0:
            self.data_set.append(a)
            self.label_set.append(torch.tensor(self.la).unsqueeze(1))
        if len(b) != 0:
            self.data_set.append(b)
            self.label_set.append(torch.tensor(self.lb).unsqueeze(1))
        if len(c) != 0:
            self.data_set.append(c)
            self.label_set.append(torch.tensor(self.lc).unsqueeze(1))
        if len(d) != 0:
            self.data_set.append(d)
            self.label_set.append(torch.tensor(self.ld).unsqueeze(1))
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



class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        
       
        self.model = model_base.BaseModel(input_size=1, num_classes=args.num_classes,
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
                y, f = self.model(data)
                src_feat, tgt_feat = f.chunk(2, dim=0)
                s_pred, t_pred = y.chunk(2, dim=0)
                
              
                data, label, clc_loss_step = I_Softmax(3, 16, s_pred, source_labels,self.device).forward()
                pre_pseudo_label = torch.argmax(t_pred, dim=-1)
                pseudo_data, pseudo_label, pseudo_loss_step = I_Softmax(2, 16, t_pred, pre_pseudo_label,self.device).forward()
                CDA_loss = MMD(source_data.size()[0], target_data.size()[0],self.device).CDA(src_feat, source_labels, tgt_feat, pre_pseudo_label)
                MDA_loss = MMD(source_data.size()[0], target_data.size()[0],self.device).MDA(src_feat, tgt_feat)
                loss = clc_loss_step + (MDA_loss + 0.1 * CDA_loss) + 0.1 * pseudo_loss_step
                        
            
                epoch_acc['Source Data']  += utils.get_accuracy(s_pred, source_labels)
                
                epoch_loss['Source Classifier'] += clc_loss_step
                epoch_loss['MDA_loss'] += MDA_loss
                epoch_loss['CDA_loss'] += CDA_loss
                epoch_loss['pseudo_loss_step'] += pseudo_loss_step

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
            if self.args.tsne:
                self.epoch = epoch
                if epoch % 50 == 0:
                #if epoch == 1 or epoch % 5 == 0:
                    self.test_tsne()
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
                pred,_ = self.model(target_data)
                acc += utils.get_accuracy(pred, target_labels)
        acc /= num_iter
        logging.info('Val-Acc Target Data: {:.4f}'.format(acc))
        return acc
    

