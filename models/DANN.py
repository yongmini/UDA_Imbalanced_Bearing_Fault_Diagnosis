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
from utils import visualize_tsne_and_confusion_matrix
import numpy as np     
from sklearn.metrics import confusion_matrix
import os
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import aug
import data_utils
    
    
class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 512
        self.model = model_base.BaseModel(input_size=1, num_classes=args.num_classes,
                                     dropout=args.dropout).to(self.device)
        self.domain_discri = model_base.ClassifierMLP(input_size=output_size, output_size=1,
                        dropout=args.dropout, last='sigmoid').to(self.device)
        grl = utils.GradientReverseLayer() 
        self.domain_adv = utils.DomainAdversarialLoss(self.domain_discri, grl=grl)
                  
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
                y_s, _ = y.chunk(2, dim=0)
        
                loss_c = F.cross_entropy(y_s, source_labels)
                loss_d, acc_d = self.domain_adv(f_s, f_t)
                loss = loss_c + tradeoff[0] * loss_d
                epoch_acc['Source Data']  += utils.get_accuracy(y_s, source_labels)
                epoch_acc['Discriminator']  += acc_d
                
                epoch_loss['Source Classifier'] += loss_c
                epoch_loss['Discriminator'] += loss_d

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
                
            #log the best model according to the val accuracy
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
             
            if self.args.tsne:
                self.epoch = epoch
                if epoch == 1 or epoch % 5 == 0:
                    self.test_tsne()
                
     
        acc=self.test()
        acc_formatted = f"{acc:.3f}"
        wandb.log({"target_acc": float(acc_formatted)})    
    

                
        
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


    
    def test_tsne(self):
        self.model.eval()
        acc = 0.0
        
        
        self.dataloaders2 = {x: torch.utils.data.DataLoader(self.datasets[x],
                                                        batch_size=64,
                                                        shuffle=False,
                                                        drop_last=False,
                                                        pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train']}

                
        
   
        iters = iter(self.dataloaders2['train'])#val
        num_iter = len(iters)
        all_features = []
        all_labels = []
        all_preds = [] 
        all_classifications = []
        with torch.no_grad():
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels, _ = next(iters)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                pred, features = self.model(target_data)
                probabilities = F.softmax(pred, dim=1)
                pred=pred.argmax(dim=1)
                all_features.append(features.cpu().numpy())
                all_labels.append(target_labels.cpu().numpy())
                all_preds.append(pred.cpu().numpy())

                # Process each sample
                for label, prediction, confidence in zip(target_labels, pred, probabilities):
                    if label != prediction:
                        # Misclassified
                        all_classifications.append(['Incorrect', label.item(), prediction.item()] + confidence.tolist())
                    else:
                        # Correctly classified
                        all_classifications.append(['Correct', label.item(), prediction.item()] + confidence.tolist())



        # Concatenate features and labels
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        
        cm = confusion_matrix(all_labels, all_preds)
        classification_filename = f"classifications_{self.args.imba}_{self.args.model_name}_{self.epoch}.txt"
        header = "Classification_Status TrueLabel PredictedLabel " + " ".join(f"Conf_Class{i}" for i in range(probabilities.shape[1]))
        np.savetxt(os.path.join(self.args.save_dir, classification_filename), np.array(all_classifications), fmt='%s', header=header)

        # Perform t-SNE and save plot
        filename = f"tsne_conmat_imba_unlabel_{str(self.args.imba)}_{self.args.model_name}_{self.epoch}.png"

        visualize_tsne_and_confusion_matrix(all_features, all_labels,all_preds, cm, self.args.save_dir,filename)
        
