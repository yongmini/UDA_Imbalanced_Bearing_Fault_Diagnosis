import os
import math
import torch
import logging
import importlib
from torch import optim
from torch.utils.data.dataset import ConcatDataset



class InitTrain(object):
    
    def __init__(self, args):
        self.args = args
        if args.cuda_device:
            self.device = torch.device("cuda:" + args.cuda_device)
            logging.info('using {} / {} gpus'.format(len(args.cuda_device.split(',')), torch.cuda.device_count()))
        else:
            self.device = torch.device("cpu")
            logging.info('using cpu')
        if args.train_mode == 'source_combine':
            self.num_source = 1
        else:
            self.num_source = len(args.source_name)
    
    
    def _get_lr_scheduler(self, optimizer):
        '''
        Get learning rate scheduler for optimizer.
        '''
        args = self.args
        assert args.lr_scheduler in ['step', 'exp', 'stepLR', 'fix'], f"lr scheduler should be 'step', 'exp', 'stepLR' or 'fix', but got {args.lr_scheduler}"
        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, args.gamma)
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
    
    
    def _init_data(self, concat_src=False, concat_all=False):
        '''
        Initialize the datasets.
        concat_src: Whether to concatenate the source datasets.
        concat_all: Whether to concatenate the source datasets and target training set.
        '''
        args = self.args
        
        self.datasets = {}
        idx = 0          
        for i, source in enumerate(args.source_name):
       
            if args.train_mode == 'multi_source':
                idx = i
            if '_' in source:
                src, condition = source.split('_')[0], int(source.split('_')[1])
                print(f"Loading source: {src}, Condition: {condition}")  # Added print for source and condition
                data_root = os.path.join(args.data_dir, src)
                Dataset = importlib.import_module("data_loader.source_load").dataset
                self.datasets[source] = Dataset(data_root, src, args.faults, args.signal_size, args.normlizetype, condition=condition
                                                ).data_preprare(source_label=idx, random_state=args.random_state)

        for key in self.datasets.keys():
            logging.info('Source set {} number of samples {}.'.format(key, len(self.datasets[key])))
          #  wandb.log({f"Source Set {key} Size": len(self.datasets[key])})
            self.datasets[key].summary()
            
        if args.imba:
            if 'SEU' in args.target:
                self.imbalance_ratio = {0:1, 1: 1, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.01, 6: 0.01, 7: 0.01, 8: 0.01, 9: 0.01}
             #   wandb.log({"imba": self.imbalance_ratio})
            elif 'JNU' in args.target:
                self.imbalance_ratio = {0:1, 1: 0.05, 2: 0.05, 3: 0.05}
            #    wandb.log({"imba": self.imbalance_ratio})
            elif 'CWRU' in args.target:
                self.imbalance_ratio = {0:1, 1: 0.2, 2: 0.2, 3: 0.2}
           #     wandb.log({"imba": self.imbalance_ratio})

        else:
            self.imbalance_ratio = None

        if '_' in args.target:
            tgt, condition = args.target.split('_')[0], int(args.target.split('_')[1])
            print(f"Setting up target: {tgt}, Condition: {condition}")  # Added print for target and condition
            data_root = os.path.join(args.data_dir, tgt)
            Dataset = importlib.import_module("data_loader.target_load").dataset
            self.datasets['train'], self.datasets['val'] = Dataset(data_root, tgt, args.faults, args.signal_size, args.normlizetype, condition=condition
                                                                ).data_preprare(source_label=idx+1, random_state=args.random_state, imbalance_ratio=self.imbalance_ratio)

    
        logging.info('target training set number of samples {}.'.format(len(self.datasets['train'])))
      #  wandb.log({"target training Set Size": len(self.datasets['train'])})
        self.datasets['train'].summary()
        logging.info('target validation set number of samples {}.'.format(len(self.datasets['val'])))
       # wandb.log({"Validation Set Size": len(self.datasets['val'])})
        self.datasets['val'].summary()
        
        dataset_keys = args.source_name + ['train', 'val']
        if concat_src:
            self.datasets['concat_source'] = ConcatDataset([self.datasets[s] for s in args.source_name])
            dataset_keys.append('concat_source')
        if concat_all:
            self.datasets['concat_all'] = ConcatDataset([self.datasets[s] for s in args.source_name]+[self.datasets['train']])
            dataset_keys.append('concat_all')

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x],
                                              batch_size=args.batch_size,
                                              shuffle=(False if x == 'val' else True),
                                              num_workers=args.num_workers, drop_last=(False if x == 'val' else True),
                                              pin_memory=(True if self.device == 'cuda' else False))
                                              for x in dataset_keys}

        
        self.iters = {x: iter(self.dataloaders[x]) for x in dataset_keys}
        
