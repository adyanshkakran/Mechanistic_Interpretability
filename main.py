
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
    parser.add_argument('--stack_depths', type=parse_list, default=[5, 15])

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

    res, model, trainer, train_outbeddings, val_outbeddings = train(model, trainer, train_loader, val_loader)

    res, _ = test(model, trainer, train_loader)
    logger.log(f'Training results: {res}')
    
    res, test_outbeddings = test(model, trainer, test_loader)
    logger.log(f'Test results: {res}')

    X_probe_train, y_probe_train = get_probe_data(train_outbeddings)
    X_probe_val, y_probe_val = get_probe_data(val_outbeddings)
    X_probe_test, y_probe_test = get_probe_data(test_outbeddings)

    train_dataset = make_dataset(X_probe_train, y_probe_train)
    val_dataset = make_dataset(X_probe_val, y_probe_val)
    test_dataset = make_dataset(X_probe_test, y_probe_test)

    train_probe_loader, val_probe_loader, test_probe_loader = get_probe_loaders(train_dataset, val_dataset, test_dataset, batch_size=args.batch_size)

    res = probe_all_models(X_probe_train.cpu().detach().numpy(), y_probe_train.cpu().detach().numpy(), X_probe_test.cpu().detach().numpy(), y_probe_test.cpu().detach().numpy())
    logger.log(f'Probe results:')
    for r in res:
        logger.log(r['model'])
        logger.log(r['cr'])
        logger.log('|==============================|')

