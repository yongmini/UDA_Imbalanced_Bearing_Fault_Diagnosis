'''
Base model: Long, M., Cao, Z., Wang, J. and Jordan, M.I., 2018. Conditional adversarial
    domain adaptation. Advances in neural information processing systems, 31.
Reference code: https://github.com/thuml/Transfer-Learning-Library
'''
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import wandb
import utils
import model_base
from train_utils import InitTrain

# "I am currently making revisions, so the final code will be released later."

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
    
class ConditionalDomainAdversarialLoss(nn.Module):
   
    def __init__(self, domain_discriminator: nn.Module,
                 randomized: bool = False, num_classes: int = -1,
                 features_dim: int = -1, randomized_dim: int = 512,
                 reduction: str = 'mean'):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = utils.GradientReverseLayer() 

        self.reduction = reduction

        if randomized:
            assert num_classes > 0 and features_dim > 0 and randomized_dim > 0
            self.map = RandomizedMultiLinearMap(512, 4, 2048)
        else:
            self.map = MultiLinearMap()
        self.bce = lambda input, target: F.binary_cross_entropy(input, target,
                                                                        reduction=reduction) 
        self.domain_discriminator_accuracy = None

    def balance_and_concatenate(self, f_s, f_t, g_s, g_t):
        g_s = F.softmax(g_s, dim=1).detach()
        g_t = F.softmax(g_t, dim=1).detach()

        if g_s.size(0) == 0 or g_t.size(0) == 0:
            raise ValueError("Input tensors g_s or g_t are empty.")

        _, labels_s = torch.max(g_s, dim=1)
        _, labels_t = torch.max(g_t, dim=1)
        all_labels = torch.cat([labels_s, labels_t]).unique()

        balanced_f_s, balanced_f_t = [], []
        balanced_g_s, balanced_g_t = [], []

        for label in all_labels:
            indices_s = torch.where(labels_s == label)[0]
            indices_t = torch.where(labels_t == label)[0]

            if len(indices_s) > 0 and len(indices_t) > 0:
                min_count = min(len(indices_s), len(indices_t))
                selected_indices_s = indices_s[:min_count]
                selected_indices_t = indices_t[:min_count]

                balanced_f_s.append(f_s[selected_indices_s])
                balanced_f_t.append(f_t[selected_indices_t])
                balanced_g_s.append(g_s[selected_indices_s])
                balanced_g_t.append(g_t[selected_indices_t])
             
        if len(balanced_f_s) == 0 or len(balanced_f_t) == 0 or len(balanced_g_s) == 0 or len(balanced_g_t) == 0:
            return f_s, f_t, g_s, g_t

        balanced_f_s = torch.cat(balanced_f_s, dim=0)
        balanced_f_t = torch.cat(balanced_f_t, dim=0)
        balanced_g_s = torch.cat(balanced_g_s, dim=0)
        balanced_g_t = torch.cat(balanced_g_t, dim=0)
      

        return balanced_f_s, balanced_f_t, balanced_g_s, balanced_g_t
    


    def forward(self, g_s: torch.Tensor, f_s: torch.Tensor, g_t: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f_s, f_t, g_s, g_t = self.balance_and_concatenate(f_s, f_t, g_s, g_t)
        f = torch.cat((f_s, f_t), dim=0)
        g = torch.cat((g_s, g_t), dim=0)
        
        g = F.softmax(g, dim=1).detach()
        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)

        d_label = torch.cat((
            torch.ones((g_s.size(0), 1)).to(g_s.device),
            torch.zeros((g_t.size(0), 1)).to(g_t.device),
        ))
        self.domain_discriminator_accuracy = utils.binary_accuracy(d, d_label)

        return F.binary_cross_entropy(d, d_label, reduction=self.reduction)


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 512
        self.domain_discri = model_base.ClassifierMLP(input_size=output_size * args.num_classes, output_size=1,
                        dropout=args.dropout, last='sigmoid').to(self.device)
        self.domain_adv = ConditionalDomainAdversarialLoss(self.domain_discri)
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
            raise Exception("This model cannot be trained with multi-source data.")

        self.optimizer = self._get_optimizer([self.model, self.domain_discri])
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
            self.domain_discri.train()
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
                f_s, f_t = f.chunk(2, dim=0)
                y_s, y_t = y.chunk(2, dim=0)
                

                loss_c = F.cross_entropy(y_s, source_labels)
                 
                local_dist = self.domain_adv(y_s, f_s, y_t, f_t)

                loss = loss_c  +  tradeoff[0]*local_dist  
 
                epoch_acc['Source Data']  += utils.get_accuracy(y_s, source_labels)
 
                epoch_acc['Discriminator']  += self.domain_adv.domain_discriminator_accuracy
                
                epoch_loss['Source Classifier'] += loss_c
                epoch_loss['Discriminator'] += local_dist

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
            
       #   @  log the best model according to the val accuracy
            new_acc = self.test()
            
            last_acc_formatted = f"{new_acc:.2f}"
            wandb.log({"last_target_acc": float(last_acc_formatted)})
            
            if new_acc >= best_acc:
                best_acc = new_acc
                best_epoch = epoch
            logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch, best_acc))
            
            best_acc_formatted = f"{best_acc:.2f}"
            wandb.log({"best_target_acc": float(best_acc_formatted)})
    
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                             
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
    
 