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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
from mpl_toolkits.mplot3d import Axes3D
import data_loader.aug as aug
import copy
class Trainset(InitTrain):
    
    def __init__(self, args):
        super(Trainset, self).__init__(args)
        output_size = 2560
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
                pred=_.argmax(dim=1)
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
            for key in epoch_loss.keys():
                logging.info('Train-Loss {}: {:.4f}'.format(key, epoch_loss[key]/num_iter))
            for key in epoch_acc.keys():
                logging.info('Train-Acc {}: {:.4f}'.format(key, epoch_acc[key]/num_iter))
                
            # log the best model according to the val accuracy
            new_acc = self.test()
            
            last_acc_formatted = f"{new_acc:.2f}"
            wandb.log({"last_target_acc": float(last_acc_formatted)})
            
            
            if new_acc >= best_acc:
                best_acc = new_acc
                best_epoch = epoch
            logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch, best_acc))
            
            best_acc_formatted = f"{best_acc:.2f}"
            wandb.log({"best_target_acc": float(best_acc_formatted)})
            
        
        if self.args.tsne:
            #  self.epoch = epoch
            if epoch == 1 or epoch % 5 == 0:
                self.test_tsne()
       
       
            
            
        self.model.train()
        num_iter = len(self.dataloaders['train'])
        
        
        class RandomAddGaussian(object):
            def __init__(self, sigma=0.01):
                self.sigma = sigma
                
            def __call__(self, seq):
                if np.random.randint(2):
                    return seq
                else:
                    noise = torch.normal(mean=0, std=self.sigma, size=seq.shape, device=seq.device)
                    return seq + noise

        train_transform = aug.Compose([
                     RandomAddGaussian()])
        def mixup_data(x, y, alpha=1.0):
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1

            batch_size = x.size()[0]
            index = torch.randperm(batch_size).to(x.device)

            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam

        def mixup_criterion(criterion, pred, y_a, y_b, lam):
            return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
        
        def get_next_batch_balanced(dataloader, iter_dict, src, model, device, max_ratio=3):
            try:
                data = next(iter_dict[src])
            except StopIteration:
                iter_dict[src] = iter(dataloader[src])
                data = next(iter_dict[src])
            
            # 모델의 예측 결과 얻기
            with torch.no_grad():
                model.eval()
                pred = model(data[0].to(device))[0].argmax(dim=1)
            
            # 예측된 레이블 분포 계산
            unique_labels, label_counts = torch.unique(pred, return_counts=True)
            max_count = label_counts.max()
            
            # 많이 예측된 레이블의 데이터 줄이기
            indices = []
            for label in unique_labels:
                label_indices = torch.where(pred == label)[0]
                if label_counts[label] > max_count / max_ratio:
                    num_samples = int(max_count / max_ratio)
                    sampled_indices = torch.randperm(len(label_indices))[:num_samples]
                    indices.append(label_indices[sampled_indices])
                else:
                    indices.append(label_indices)
            
            indices = torch.cat(indices)
            data = (data[0][indices], data[1][indices])
            pred = pred[indices]
            
            return data[0].to(device), pred.to(device)
        
        original_model = copy.deepcopy(self.model)
        original_model.eval()
        self.model.train()
        for epo in range(1):
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_pred = get_next_batch_balanced(self.dataloaders, self.iters, 'train', original_model, self.device)
                source_data, source_labels = get_next_batch_balanced(self.dataloaders, self.iters, src, original_model, self.device)
                
                # forward
                data = torch.cat((source_data, target_data), dim=0)
                
                with torch.no_grad():
                    y, f = original_model(data)
                    f_s, f_t = f.chunk(2, dim=0)
                    y_s, _ = y.chunk(2, dim=0)
                
                self.optimizer.zero_grad()
                
                target_data = train_transform(target_data)
                target_pred = target_pred[:target_data.size(0)]  # target_pred의 크기를 target_data에 맞춤
                mixed_target_data, target_pred_a, target_pred_b, lam = mixup_data(target_data, target_pred)
                
                y, _ = self.model(mixed_target_data)
                
                loss_c = mixup_criterion(F.cross_entropy, y, target_pred_a, target_pred_b, lam)
                
                
              #  loss_d, acc_d = self.domain_adv(f_s, f_t)
                loss = loss_c #+ tradeoff[0] * loss_d
                # epoch_acc['Source Data']  += utils.get_accuracy(y_s, source_labels)
                # epoch_acc['Discriminator']  += acc_d
                
                # epoch_loss['Source Classifier'] += loss_c
                # epoch_loss['Discriminator'] += loss_d

                # backward
                loss.backward()
                self.optimizer.step()    
            
        self.test()   
        self.test_tsne()
        # if self.args.tsne:
        #         self.test_tsne()
        #       #  self.test_tsne_all()
            
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
    
    
    def test_tsne(self):
        self.model.eval()
        acc = 0.0
        
        
        self.dataloaders2 = {x: torch.utils.data.DataLoader(self.datasets[x],
                                                        batch_size=64,
                                                        shuffle=False,
                                                        drop_last=False,
                                                        pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['val']}

                
        
   
        iters = iter(self.dataloaders2['val'])#val
        num_iter = len(iters)
        all_features = []
        all_labels = []
        all_preds = [] 
        with torch.no_grad():
            for i in tqdm(range(num_iter), ascii=True):
                target_data, target_labels, _ = next(iters)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                pred, features = self.model(target_data)
                
                pred=pred.argmax(dim=1)
                all_features.append(features.cpu().numpy())
                all_labels.append(target_labels.cpu().numpy())
                all_preds.append(pred.cpu().numpy())

        # Concatenate features and labels
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        
        cm = confusion_matrix(all_labels, all_preds)

        # Perform t-SNE and save plot
       # filename = f"tsne_conmat_imba_{str(self.args.imba)}_{self.args.model_name}_{self.epoch}.png"
        filename = f"tsne_conmat_imba_{str(self.args.imba)}_{self.args.model_name}.png"
        visualize_tsne_and_confusion_matrix(all_features, all_labels,all_preds, cm, self.args.save_dir,filename)
        
        
    # def test_tsne_all(self):
    #     self.model.eval()
    #     source_iter = iter(self.dataloaders[self.args.source_name[0]])  # Source data iterator 추가
    #     target_iter = iter(self.dataloaders['val'])
        
    #     all_features = []
    #     all_labels = []
    #     all_domains = []  # Source인지 Target인지를 구분하는 레이블 추가
        
    #     with torch.no_grad():
    #         for _ in range(len(target_iter)):
    #             source_data, source_labels ,_ = next(source_iter)
    #             target_data, target_labels, _ = next(target_iter)
                
    #             source_data, source_labels = source_data.to(self.device), source_labels.to(self.device)
    #             target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                
    #             _, source_features = self.model(source_data)
    #             _, target_features = self.model(target_data)
                
    #             all_features.append(source_features.cpu().numpy())
    #             all_features.append(target_features.cpu().numpy())
                
    #             all_labels.append(source_labels.cpu().numpy())
    #             all_labels.append(target_labels.cpu().numpy())
                
    #             all_domains.append(np.zeros_like(source_labels.cpu().numpy()))  # Source는 0
    #             all_domains.append(np.ones_like(target_labels.cpu().numpy()))   # Target은 1
        
    #     all_features = np.concatenate(all_features, axis=0)
    #     all_labels = np.concatenate(all_labels, axis=0)
    #     all_domains = np.concatenate(all_domains, axis=0)
        
        
        
    #     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    #     features_tsne = tsne.fit_transform(all_features)
        
    #     # 시각화를 위한 색상 맵 설정
    #     num_classes = len(np.unique(all_labels))
    #     colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
        
    #     # Source와 Target에 대한 t-SNE 그래프 그리기
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     markers = ['o', 'x']  # Source는 'o', Target은 'x'로 표시
    #     for i, domain in enumerate(['Source', 'Target']):
    #         mask = all_domains == i
    #         for j, label in enumerate(np.unique(all_labels)):
    #             label_mask = all_labels[mask] == label
    #             ax.scatter(features_tsne[mask][label_mask, 0], features_tsne[mask][label_mask, 1],
    #                     color=colors[j], marker=markers[i], label=f'{domain} - Class {label}', alpha=0.8)
        
        
    #     ax.set_xlabel('t-SNE Feature 1')
    #     ax.set_ylabel('t-SNE Feature 2')
    #     ax.set_title('t-SNE Visualization of Source and Target Domains')
    #     ax.legend()
        
    #     # t-SNE 그래프 저장
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.args.save_dir, 'domain_tsne.png'), dpi=300)
    #     plt.close()
            
    def test_tsne_all(self):
        self.model.eval()
        source_iter = iter(self.dataloaders[self.args.source_name[0]])  # Source data iterator 추가
        target_iter = iter(self.dataloaders['val'])
        
        all_features = []
        all_labels = []
        all_domains = []  # Source인지 Target인지를 구분하는 레이블 추가
        
        with torch.no_grad():
            for _ in range(len(target_iter)):
                source_data, source_labels, _ = next(source_iter)
                target_data, target_labels, _ = next(target_iter)
                
                source_data, source_labels = source_data.to(self.device), source_labels.to(self.device)
                target_data, target_labels = target_data.to(self.device), target_labels.to(self.device)
                
                _, source_features = self.model(source_data)
                _, target_features = self.model(target_data)
                
                all_features.append(source_features.cpu().numpy())
                all_features.append(target_features.cpu().numpy())
                
                all_labels.append(source_labels.cpu().numpy())
                all_labels.append(target_labels.cpu().numpy())
                
                all_domains.append(np.zeros_like(source_labels.cpu().numpy()))  # Source는 0
                all_domains.append(np.ones_like(target_labels.cpu().numpy()))   # Target은 1
        
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_domains = np.concatenate(all_domains, axis=0)
        
        tsne = TSNE(n_components=3, perplexity=30, random_state=42)  # n_components를 3으로 설정
        features_tsne = tsne.fit_transform(all_features)
        
        # 시각화를 위한 색상 맵 설정
        num_classes = len(np.unique(all_labels))
        colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
        
        # Source와 Target에 대한 3D t-SNE 그래프 그리기
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')  # 3D 그래프를 그리기 위한 Axes3D 사용
        markers = ['o', 'x']  # Source는 'o', Target은 'x'로 표시
        for i, domain in enumerate(['Source', 'Target']):
            mask = all_domains == i
            for j, label in enumerate(np.unique(all_labels)):
                label_mask = all_labels[mask] == label
                ax.scatter(features_tsne[mask][label_mask, 0], features_tsne[mask][label_mask, 1], features_tsne[mask][label_mask, 2],
                        color=colors[j], marker=markers[i], label=f'{domain} - Class {label}', alpha=0.8)
        
        ax.set_xlabel('t-SNE Feature 1')
        ax.set_ylabel('t-SNE Feature 2')
        ax.set_zlabel('t-SNE Feature 3')  # z축 레이블 추가
        ax.set_title('3D t-SNE Visualization of Source and Target Domains')
        ax.legend()
        filename = f'domain_tsne_3d_imba_{self.args.imba}.png'
        # 3D t-SNE 그래프 저장
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, filename), dpi=300)
        plt.close()
            