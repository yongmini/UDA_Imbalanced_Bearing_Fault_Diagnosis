'''
Paper: Maximum mean square discrepancy: A new discrepancy representation metric for mechanical fault transfer diagnosis
Reference code: https://github.com/QinYi-team/MMSD/blob/main/MMSD.ipynb
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


class MMSDLoss(nn.Module):

    def __init__(self, sigmas=(1,), wts=None, biased=True):
        super(MMSDLoss, self).__init__()
        self.sigmas = sigmas
        self.wts = wts if wts is not None else [1] * len(sigmas)
        self.biased = biased

    def forward(self, X, Y):
        K_XX, K_XY, K_YY, d = self._mix_rbf_kernel(X, Y, self.sigmas, self.wts)
        mmsd_value = self._mmsd(K_XX, K_XY, K_YY, const_diagonal=d, biased=self.biased)
        return mmsd_value

    def _mix_rbf_kernel(self, X, Y, sigmas, wts):
        XX = torch.matmul(X, X.T)
        XY = torch.matmul(X, Y.T)
        YY = torch.matmul(Y, Y.T)

        X_sqnorms = XX.diag()
        Y_sqnorms = YY.diag()

        r = lambda x: x.unsqueeze(0)
        c = lambda x: x.unsqueeze(1)

        K_XX, K_XY, K_YY = 0, 0, 0
        for sigma, wt in zip(sigmas, wts):
            gamma = 1 / (2 * sigma**2)
            K_XX += wt * torch.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
            K_XY += wt * torch.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
            K_YY += wt * torch.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))
        return K_XX, K_XY, K_YY, torch.sum(torch.tensor(wts))

    def _mmsd(self, K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
        m = K_XX.size(0)
        n = K_YY.size(0)

        C_K_XX = K_XX**2
        C_K_YY = K_YY**2
        C_K_XY = K_YY**2

        if biased:
            mmsd = (C_K_XX.sum() / (m * m) + C_K_YY.sum() / (n * n) - 2 * C_K_XY.sum() / (m * n))
        else:
            if const_diagonal is not False:
                trace_X = m * const_diagonal
                trace_Y = n * const_diagonal
            else:
                trace_X = torch.trace(C_K_XX)
                trace_Y = torch.trace(C_K_YY)

            mmsd = ((C_K_XX.sum() - trace_X) / ((m - 1) * m)
                + (C_K_YY.sum() - trace_Y) / ((n - 1) * n)
                - 2 * C_K_XY.sum() / (m * n))
        
        return mmsd



class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        
        self.mmsd = MMSDLoss()
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
                pred, _ = y.chunk(2, dim=0)
                
                loss_penalty = self.mmsd(src_feat, tgt_feat)
                loss_c = F.cross_entropy(pred, source_labels)
                loss = loss_c + tradeoff[0] * loss_penalty
                epoch_acc['Source Data']  += utils.get_accuracy(pred, source_labels)
                
                epoch_loss['Source Classifier'] += loss_c
                epoch_loss['MMSD'] += loss_penalty

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