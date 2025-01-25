import os
import math
import torch
import logging
import importlib
from torch import optim


class InitTrain(object):
    
    def __init__(self, args):
        self.args = args
        if args.cuda_device:
            self.device = torch.device("cuda:" + args.cuda_device)
            logging.info('using {} / {} gpus'.format(len(args.cuda_device.split(',')), torch.cuda.device_count()))
        else:
            self.device = torch.device("cpu")
            logging.info('using cpu')

    
    def _get_lr_scheduler(self, optimizer):
        '''
        Get learning rate scheduler for optimizer.
        '''
        args = self.args
        assert args.lr_scheduler in ['step', 'exp', 'stepLR','cosine', 'fix'], f"lr scheduler should be 'step', 'exp', 'stepLR' or 'fix', but got {args.lr_scheduler}"
        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'cosine':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0)
        elif args.lr_scheduler == 'fix':
            lr_scheduler = None
            
        return lr_scheduler
    
    
    def _get_optimizer(self, model):
        '''
        Get optimizer for model.
        '''
        args = self.args
        if type(model) == list:
            par =  [{'params': md.parameters()} for md in model]
        else:
            par = model.parameters()
        
        # Define the optimizer
        assert args.opt in ['sgd', 'adam'], f"optimizer should be 'sgd' or 'adam', but got {args.opt}"
        if args.opt == 'sgd':
            optimizer = optim.SGD(par, lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            optimizer = optim.Adam(par, lr=args.lr, betas=args.betas,
                                   weight_decay=args.weight_decay)
        
        return optimizer
    
    
    def _get_tradeoff(self, tradeoff_list, epoch=None):
        '''
        Get trade-off parameters for loss.
        '''

        tradeoff = []
        for item in tradeoff_list:      
            if item == 'exp':
                tradeoff.append(2 / (1 + math.exp(-10 * (epoch-1) / (self.args.max_epoch-1))) - 1)
            elif type(item) == float or type(item) == int:
                tradeoff.append(item)
            else:
                raise Exception(f"unknown trade-off type {item}")
        
        return tradeoff
        
        
    def _init_data(self):
        '''
        Initialize datasets and data loaders.
        '''

        args = self.args 


        if args.imba:
   

            if args.imba == 'B2B':
                self.tr_imba_ratio = None
                self.te_imba_ratio = None
                

            elif args.imba == 'B2I':
           
                if 'JNU' in args.target:
                    self.tr_imba_ratio = None
                    self.te_imba_ratio = {0: 1, 1: args.imbalance_ratio1, 2: args.imbalance_ratio2, 3: args.imbalance_ratio3}

            elif args.imba == 'I2I':

                if 'JNU' in args.target:
                    self.tr_imba_ratio = {0: 1, 1: args.imbalance_ratio3, 2: args.imbalance_ratio2, 3: args.imbalance_ratio1}
                    self.te_imba_ratio = {0: 1, 1: args.imbalance_ratio1, 2: args.imbalance_ratio2, 3: args.imbalance_ratio3}

        self.datasets = {} 

        
        src, condition = args.source_name[0].split('_')[0], int(args.source_name[0].split('_')[1])  
        print(f"Loading source: {src}, Condition: {condition}")  
        data_root = os.path.join(args.data_dir, src) 
        Dataset = importlib.import_module("data_loader.source_load").dataset  
 
        self.datasets['source_train'], self.datasets['source_val'], class_count = Dataset(data_root, src, args.faults, args.signal_size, args.normlizetype, condition=condition
                                                                ).data_prepare(self.tr_imba_ratio,args.random_state)
        self.class_count = class_count.tolist()
        

        logging.info('source training set number of samples {}.'.format(len(self.datasets['source_train'])))
        logging.info('source validation set number of samples {}.'.format(len(self.datasets['source_val'])))

        
        tgt, condition = args.target.split('_')[0], int(args.target.split('_')[1])  
        print(f"Setting up target: {tgt}, Condition: {condition}")  
        data_root = os.path.join(args.data_dir, tgt)  
        Dataset = importlib.import_module("data_loader.target_load").dataset 
        
        self.datasets['target_train'], self.datasets['target_val'] = Dataset(data_root, tgt, args.faults, args.signal_size, args.normlizetype, condition=condition
                                                                ).data_prepare(self.te_imba_ratio,args.random_state)


        logging.info('target training set number of samples {}.'.format(len(self.datasets['target_train'])))
        logging.info('target validation set number of samples {}.'.format(len(self.datasets['target_val'])))

        dataset_keys = ['source_train', 'source_val', 'target_train', 'target_val']
        
        # 데이터 로더 설정
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x],
                                            batch_size=args.batch_size,
                                            shuffle=(True if x.split('_')[1] == 'train' else False),    
                                            num_workers=args.num_workers, 
                                            drop_last=(True if x.split('_')[1] == 'train' else False),  
                                            pin_memory=(True if self.device == 'cuda' else False))
                                            for x in dataset_keys}

        self.iters = {x: iter(self.dataloaders[x]) for x in dataset_keys}  
