import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import utils
from train_utils import InitTrain
import torch.nn as nn
from models.resnet1d import feature_extractor


class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)

        self._init_data()

        self.G = feature_extractor().to(self.device)
 
        self.classifier_layer = nn.Linear(512, args.num_classes).to(self.device)   


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

        if args.opt=='sgd':
            self.optimizer = torch.optim.SGD([
                {'params': self.G.parameters(), 'lr': args.lr}, 
                {'params': self.classifier_layer.parameters(), 'lr': args.lr}
            ], lr=args.lr, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam([
                {'params': self.G.parameters(), 'lr': args.lr}, 
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
        
            num_iter = len(self.dataloaders['source_train'])               
            for i in tqdm(range(num_iter), ascii=True):
                source_data, source_labels = utils.get_next_batch(self.dataloaders,
                                            self.iters, 'source_train', self.device)
                # forward
                self.optimizer.zero_grad()
                
                pred = self.classifier_layer(self.G(source_data))
                loss = F.cross_entropy(pred, source_labels)
                epoch_acc['Source Data']  += utils.get_accuracy(pred, source_labels)
                
                epoch_loss['Source Classifier'] += loss

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
        self.classifier_layer.eval()
        source_acc = 0.0
        iters = iter(self.dataloaders['source_val'])

        
        num_iter = len(iters)
        with torch.no_grad():
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels = next(iters)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                pred = self.classifier_layer(self.G(target_data))
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
                pred =  self.classifier_layer(self.G(target_data))
                acc += utils.get_accuracy(pred, target_labels)

        acc /= num_iter
        logging.info('Val-Acc Target Data: {:.4f}'.format(acc))


        return source_acc,acc
        