import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import utils
from train_utils import InitTrain
import numpy as np     
from loss.adversarial_loss import GradientReverseLayer, DomainAdversarialLoss
import torch.nn as nn
from models.resnet1d import feature_extractor

np.seterr(divide='ignore')

class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 512
        self.use_bottleneck = args.use_bottleneck

        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(output_size, 256),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list).to(self.device)
            self.feature_dim = 256
        else:
            self.feature_dim = output_size
 

        self.G = feature_extractor().to(self.device)
        domain_discri_list = [
                nn.Linear(self.feature_dim , int(self.feature_dim/2)),
                nn.ReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(int(self.feature_dim/2), 1),
                nn.Sigmoid()
            ]
        self.domain_discri = nn.Sequential(*domain_discri_list).to(self.device)
        
        grl = GradientReverseLayer() 
        self.domain_adv = DomainAdversarialLoss(self.domain_discri, grl=grl)        
        
        self.classifier_layer = nn.Linear(self.feature_dim, args.num_classes).to(self.device)   


        self._init_data()
        
    def save_model(self):
        torch.save({
            'G': self.G.state_dict(),
            'Fs': self.bottleneck_layer.state_dict(),
            'Cs': self.classifier_layer.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.G.load_state_dict(ckpt['G'])
        self.bottleneck_layer.load_state_dict(ckpt['Fs'])
        self.classifier_layer.load_state_dict(ckpt['Cs'])
        
    def train(self):
        args = self.args

        if self.use_bottleneck: 
            self.optimizer = self._get_optimizer([self.G,self.domain_discri ,self.classifier_layer,self.bottleneck_layer])
        else:
            self.optimizer = self._get_optimizer([self.G,self.domain_discri,self.classifier_layer])

        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

   
        for epoch in range(1, args.max_epoch+1):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-'*5)
            
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            epoch_acc = defaultdict(float)
            if args.use_bottleneck:
                self.G.train()
                self.bottleneck_layer.train()
                self.classifier_layer.train()
                self.domain_discri.train()
            else:
                self.G.train()
                self.classifier_layer.train()
                self.domain_discri.train()
  
            epoch_loss = defaultdict(float)
            tradeoff = self._get_tradeoff(args.tradeoff, epoch) 
            


            num_iter = len(self.dataloaders['target_train'])
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels = utils.get_next_batch(self.dataloaders,
                						 self.iters, 'target_train', self.device)
                source_data, source_labels = utils.get_next_batch(self.dataloaders,
            						     self.iters, 'source_train', self.device)
                # forward
                
                self.optimizer.zero_grad()
                data = torch.cat((source_data, target_data), dim=0)
                
                f = self.G(data)
                if self.use_bottleneck:
                    f = self.bottleneck_layer(f)
                y = self.classifier_layer(f)
                f_s, f_t = f.chunk(2, dim=0)
                y_s, y_t = y.chunk(2, dim=0)
        
     

                loss_c = F.cross_entropy(y_s, source_labels)
                loss_d, acc_d = self.domain_adv(f_s, f_t)
                loss = loss_c +  tradeoff[0]* loss_d 
                epoch_acc['Source Data']  += utils.get_accuracy(y_s, source_labels)
                epoch_acc['Discriminator']  += acc_d

                epoch_loss['Discriminator'] += loss_d 
  

                # backwardss
                loss.backward()
                self.optimizer.step()
                            
            for key in epoch_acc.keys():
                avg_acc = epoch_acc[key] / num_iter
                logging.info('Train-Acc {}: {:.4f}'.format(key, avg_acc))

            for key in epoch_loss.keys():
                logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key]/num_iter))
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
       

        self.test()

        
    def test(self):
        self.G.eval()
        if self.use_bottleneck:
            self.bottleneck_layer.eval()
        self.classifier_layer.eval()
        source_acc = 0.0
        iters = iter(self.dataloaders['source_val'])

        
        num_iter = len(iters)
        with torch.no_grad():
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels = next(iters)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                f = self.G(target_data)
                if self.use_bottleneck:
                    f = self.bottleneck_layer(f)
                pred = self.classifier_layer(f)
                source_acc += utils.get_accuracy(pred, target_labels)

        source_acc /= num_iter
        logging.info('Val-Acc source Data: {:.4f}'.format(source_acc))

        acc = 0.0

        iters = iter(self.dataloaders['target_val'])
        num_iter = len(iters)
        with torch.no_grad():
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels = next(iters)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                f = self.G(target_data)
                if self.use_bottleneck:
                    f = self.bottleneck_layer(f)
                pred = self.classifier_layer(f)
                acc += utils.get_accuracy(pred, target_labels)

        acc /= num_iter
        logging.info('Val-Acc Target Data: {:.4f}'.format(acc))

        return source_acc,acc
        
