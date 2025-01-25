import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_bottleneck', type=str2bool, default=True)
    parser.add_argument('--imba', type=str, choices=['B2B', 'B2I', 'I2I'], default='B2I', help='Imbalance setting')
    parser.add_argument('--imbalance_ratio1', type=float, default=0.15, help='Imbalance ratio1')
    parser.add_argument('--imbalance_ratio2', type=float, default=0.1, help='Imbalance ratio2')
    parser.add_argument('--imbalance_ratio3', type=float, default=0.05, help='Imbalance ratio3')
    parser.add_argument('--model_name', type=str, default='CNN', help='Name of the model to use')
    parser.add_argument('--source', type=str, default='JNU_1', 
                        help='Source data, separated by "," (select specific conditions of the dataset with name_number, such as JNU_1)')
    parser.add_argument('--target', type=str, default='JNU_0', 
                        help='Target data (select specific conditions of the dataset with name_number, such as JNU_0)')

    
    # Data and environment settings
    parser.add_argument('--data_dir', type=str, default="./dataset", help='Directory of the datasets')
    parser.add_argument('--cuda_device', type=str, default='0', help='GPU device to use (empty string for CPU)')
    parser.add_argument('--save_dir', type=str, default='./ckpt', help='Directory to save logs and model checkpoints')
    parser.add_argument('--max_epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--signal_size', type=int, default=2048, help='Signal length split by sliding window')
    parser.add_argument('--random_state', type=int, default=15, help='Random state for the entire training')

    # Optimization information
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '-1-1', 'mean-std', 'None'], default='mean-std', 
                        help='Data normalization methods')
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate') 
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for sgd')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='Betas for adam')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for both sgd and adam')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR','cosine', 'fix'], default='stepLR',
                        help='Type of learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='Parameter for the learning rate scheduler (except "fix")')
    parser.add_argument('--steps', type=str, default='30',
                        help='Step of learning rate decay for "step" and "stepLR"')
    parser.add_argument('--tradeoff', type=list, default=['exp', 'exp', 'exp'],
                        help='Trade-off coefficients for the sum of losses, integer or "exp" ("exp" represents an increase from 0 to 1)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout layer coefficient')
    
    # Save and load
    parser.add_argument('--save', type=bool, default=False, help='Save logs and trained model checkpoints')
    parser.add_argument('--load_path', type=str, default='', help='Path to the model weight file for testing without training')

    args = parser.parse_args()
    return args
