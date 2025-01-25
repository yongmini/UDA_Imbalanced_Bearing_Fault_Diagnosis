import os
import sys
sys.path.extend(['./models', './data_loader'])
import torch
import logging
import importlib
from datetime import datetime
from opt import parse_args
import numpy as np
import random

def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    
    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    return logger


def creat_file(args):
    # prepare the saving path for the model
    source = ''
    for src in args.source_name:
        source += src
    file_name = '[' + source + ']' + 'To' + '[' +\
            args.target + ']' + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.save_dir, args.model_name)
    args.save_dir= save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_path = os.path.join(save_dir, file_name)
    
    # set the logger
    logger = setlogger(args.save_path  + '.log')

    # save the args
    for k, v in args.__dict__.items():
        if k != 'source_name':
            logging.info("{}: {}".format(k, v))
    return logger, args


if __name__ == '__main__':
    
    os.environ['NUMEXPR_MAX_THREADS'] = '8'
    args = parse_args()
    
    if args.random_state is not None:
        os.environ['PYTHONHASHSEED'] = str(args.random_state)
        np.random.seed(args.random_state)
        random.seed(args.random_state)
        torch.manual_seed(args.random_state)
        torch.cuda.manual_seed(args.random_state)
        torch.cuda.manual_seed_all(args.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



    args.source_name = [x.strip() for x in list(args.source.split(','))]
    if '' in args.source_name:
        args.source_name.remove('')

    # training
    logger, args = creat_file(args)
    tgt, condition = args.target.split('_')[0], int(args.target.split('_')[1])             
    data_root = os.path.join(args.data_dir, tgt)                                           
    args.faults = sorted(os.listdir(os.path.join(data_root, 'condition_%d' % condition)))  
    args.num_classes = len(args.faults)                                                    

        
    logging.info('Detect {} classes: {}'.format(args.num_classes, args.faults)) 
    trainer = importlib.import_module(f"models.{args.model_name}").Trainset(args)          

    if args.load_path:
        trainer.load_model()
        trainer.test()
        os.remove(args.save_path + '.log')
    else:
        trainer.train()
        if args.save:
            trainer.save_model()
        else:
            os.remove(args.save_path + '.log')
    logger.handlers.clear()
