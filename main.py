
import pytorch_lightning as L
from data_loading import load_data, get_loaders, get_probe_data, get_probe_loaders, make_dataset
from transformer_predictor import TransformerPredictor, train, test
from probe import probe_all_models
from time import time
import argparse
import os

class Logger:
    def __init__(self):
        self.logs = []
        self.file_name = f'logs/logs_{int(time())}.txt'
        # check if logs directory exists
        if not os.path.exists('logs'):
            os.makedirs('logs')
        # initialize file
        with open(self.file_name, 'w') as f:
            f.write('')
    
    def log(self, msg):
        self.logs.append(msg)
        with open(self.file_name, 'a') as f:
            f.write(msg + '\n')

def parse_list(input_str):
    return input_str.split(',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--transformer_epochs', type=int, default=10)
    parser.add_argument('--data-path', type=str, default='brackets.csv')

    args = parser.parse_args()
    
    logger = Logger()
    logger.log(f'Arguments: {args}')
    
    dataset = load_data(args.data_path)

    train_loader, val_loader, test_loader = get_loaders(dataset, batch_size=args.batch_size)

    model = TransformerPredictor(
        input_dim=4,
        model_dim=args.model_dim,
        num_classes=2,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr,
        warmup=args.warmup,
        max_iters=args.max_iters,
    )
    trainer = L.Trainer(max_epochs=args.transformer_epochs, devices=1)

    res, model, trainer, _, _ = train(model, trainer, train_loader, val_loader)

    res, _, _ = test(model, trainer, train_loader)
    logger.log(f'Training results: {res}')
    
    res, _, _ = test(model, trainer, test_loader)
    logger.log(f'Test results: {res}')
