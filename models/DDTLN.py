'''
Paper: Qian, Q., Qin, Y., Luo, J., Wang, Y., and Wu, F. (2023). 
Deep discriminative transfer learning network for cross-machine fault diagnosis. 
Mechanical Systems and Signal Processing, 186, 109884.
https://github.com/QinYi-team/Code/blob/master/DDTLN/DDTLN%20code.ipynb
'''
import torch
import logging
from tqdm import tqdm
from collections import defaultdict
import utils
import model_base
from train_utils import InitTrain
from loss.distance_loss import DDM
from loss.classification_loss import I_Softmax
import torch.nn as nn
from models.resnet1d import feature_extractor

class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 512
        self.G = feature_extractor().to(self.device)
        self.classifier_layer = nn.Linear(output_size, args.num_classes).to(self.device)   

        self._init_data()
    
    def save_model(self):
        torch.save({
            'G': self.G.state_dict(),
            'Cs': self.classifier_layer.state_dict()
            }, self.args.save_path + '.pth')
        logging.info('Model saved to {}'.format(self.args.save_path + '.pth'))
    
    def load_model(self):
        logging.info('Loading model from {}'.format(self.args.load_path))
        ckpt = torch.load(self.args.load_path)
        self.G.load_state_dict(ckpt['G'])
        self.classifier_layer.load_state_dict(ckpt['Cs'])
        
        
    def train(self):
        args = self.args
        
 
        self.optimizer = torch.optim.Adam([
            {'params': self.G.parameters(), 'lr':  args.lr}, 
            {'params': self.classifier_layer.parameters(), 'lr': args.lr} 
        ], lr=args.lr)
            
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        for epoch in range(1, args.max_epoch+1):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-'*5)
            
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            epoch_acc = defaultdict(float)
  
            self.G.train()
            self.classifier_layer.train()

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

                y = self.classifier_layer(f)
                f_s, f_t = f.chunk(2, dim=0)
                y_s, y_t = y.chunk(2, dim=0)
                
              
                _, _, clc_loss_step = I_Softmax(3, 16,self.class_count, y_s, source_labels,self.device).forward()
                pre_pseudo_label = torch.argmax(y_t, dim=-1)
                pseudo_data, pseudo_label, pseudo_loss_step = I_Softmax(3, 16,self.class_count, y_t, pre_pseudo_label,self.device).forward()
                CDA_loss = DDM(source_data.size()[0], target_data.size()[0],self.device).CDA(f_s, source_labels, f_t, pre_pseudo_label)
                MDA_loss = DDM(source_data.size()[0], target_data.size()[0],self.device).MDA(f_s, f_t)
                loss = clc_loss_step + tradeoff[0]*(MDA_loss + 0.1 * CDA_loss) + 0.1 * pseudo_loss_step
                        
            
                epoch_acc['Source Data']  += utils.get_accuracy(y_s, source_labels)
                
                epoch_loss['Source Classifier'] += clc_loss_step
                epoch_loss['MDA_loss'] += MDA_loss
                epoch_loss['CDA_loss'] += CDA_loss

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
            self.classifier_layer.eval()
            source_acc = 0.0
            iters = iter(self.dataloaders['source_val'])

            
            num_iter = len(iters)
            with torch.no_grad():
                for i in tqdm(range(num_iter), ascii=True):
                    target_data, target_labels = next(iters)
                    target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                    f = self.G(target_data)
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
                    pred = self.classifier_layer(f)
                    acc += utils.get_accuracy(pred, target_labels)

            acc /= num_iter
            logging.info('Val-Acc Target Data: {:.4f}'.format(acc))

            return source_acc,acc
        
