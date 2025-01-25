import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import utils
from train_utils import InitTrain
import numpy as np     
from loss.adversarial_loss import GradientReverseLayer
import torch.nn as nn
from models.resnet1d import feature_extractor

class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 512


        self.use_bottleneck = args.use_bottleneck
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(output_size, int(output_size/2)),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list).to(self.device)
            self.feature_dim = int(output_size/2)
        else:
            self.feature_dim = output_size


        domain_discri_list = [
                nn.Linear(self.feature_dim * args.num_classes , int(self.feature_dim/2)),
                nn.ReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(int(self.feature_dim/2), 2),
            ]
        self.domain_discri = nn.Sequential(*domain_discri_list).to(self.device)

  
        self.grl = GradientReverseLayer()
        self.dist_beta = torch.distributions.beta.Beta(1., 1.)
        self.G = feature_extractor().to(self.device)
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
                batch_size = source_data.shape[0]
                self.optimizer.zero_grad()
                data = torch.cat((source_data, target_data), dim=0)
                
                f = self.G(data)
                if self.use_bottleneck:
                    f = self.bottleneck_layer(f)
                y = self.classifier_layer(f)
                f_s, f_t = f.chunk(2, dim=0)
                y_s, y_t = y.chunk(2, dim=0)
                
                loss_c = F.cross_entropy(y_s, source_labels)
                
                softmax_output_src = F.softmax(y_s, dim=-1)
                softmax_output_tgt = F.softmax(y_t, dim=-1)
               
                lmb = self.dist_beta.sample((batch_size, 1)).to(self.device)
                labels_dm = torch.concat((torch.ones(batch_size, dtype=torch.long),
                      torch.zeros(batch_size, dtype=torch.long)), dim=0).to(self.device)
        
                idxx = np.arange(batch_size)
                np.random.shuffle(idxx)
                f_s = lmb * f_s + (1.-lmb) * f_s[idxx]
                f_t = lmb * f_t + (1.-lmb) * f_t[idxx]
    
                softmax_output_src = lmb * softmax_output_src + (1.-lmb) * softmax_output_src[idxx]
                softmax_output_tgt = lmb * softmax_output_tgt + (1.-lmb) * softmax_output_tgt[idxx]
                                             
                feat_src_ = torch.bmm(softmax_output_src.unsqueeze(2),
                                     f_s.unsqueeze(1)).view(batch_size, -1)
                feat_tgt_ = torch.bmm(softmax_output_tgt.unsqueeze(2),
                                     f_t.unsqueeze(1)).view(batch_size, -1)
    
                feat = self.grl(torch.concat((feat_src_, feat_tgt_), dim=0))
                logits_dm = self.domain_discri(feat)
                loss_dm = F.cross_entropy(logits_dm, labels_dm)
                loss = loss_c + tradeoff[0] * loss_dm
                epoch_acc['Source Data']  += utils.get_accuracy(y_s, source_labels)
                
                epoch_loss['Source Classifier'] += loss_c
                epoch_loss['Discriminator'] += loss_dm

                # backward
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