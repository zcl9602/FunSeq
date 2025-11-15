import torch
from bin.modelPE_C import CATG, GenoDataset
from bin.train import main_worker
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train CATG model')
    parser.add_argument('--train_data', type=str, required=True, help='path to train data')
    parser.add_argument('--valid_data', type=str, required=True, help='path to valid data')
    parser.add_argument('--output_prefix', type=str, required=True, help='prefix of output files')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='learning rate, default=2e-3')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs, default=100')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size, default=128')
    parser.add_argument('--num_workers', type=int, default=20, help='number of workers, default=20')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop count, default=10')
    parser.add_argument('--early_stop_delta', type=float, default=0, help='early stop delta, default=0')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binary classification, default=0.5')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha for focal loss, default=0.5')
    parser.add_argument('--gamma', type=float, default=0, help='gamma for focal loss, default=0')
    parser.add_argument('--rank', type=int, default=0, help='rank of GPU, default=0')

    args = parser.parse_args()
    return args

def main(args):
    train_data = torch.load(args.train_data)
    valid_data = torch.load(args.valid_data)
    train_data = GenoDataset(train_data['x'].float(), train_data['y'].float())
    valid_data = GenoDataset(valid_data['x'].float(), valid_data['y'].float())

    print(f'train set size: {len(train_data)}\nvalid set size: {len(valid_data)}\n')
    # get shape of input data
    seq_len, _ = train_data[0][0].shape
    
    num_epochs = args.num_epochs
    num_workers = args.num_workers
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    output_prefix = args.output_prefix
    early_stop=args.early_stop
    early_stop_delta=args.early_stop_delta
    threshold=args.threshold
    alpha=args.alpha
    gamma=args.gamma
    rank=args.rank

    model = CATG()

    auroc = main_worker(model, train_data, valid_data,
                output_dir=output_prefix,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                num_workers=num_workers,
                early_stop=early_stop,
                early_stop_delta=early_stop_delta,
                rank=rank,
                threshold=threshold,
                alpha=alpha,
                gamma=gamma)
    return auroc

if __name__ == '__main__':
    args = parse_args()
    auroc = main(args)
