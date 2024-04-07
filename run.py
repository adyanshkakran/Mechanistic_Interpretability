import subprocess

def run(args):
    subprocess.run([
        'python3',
        'main.py',
        '--model_dim', str(args.model_dim),
        '--num_heads', str(args.num_heads),
        '--num_layers', str(args.num_layers),
        '--lr', str(args.lr),
        '--warmup', str(args.warmup),
        '--max_iters', str(args.max_iters),
        '--batch_size', str(args.batch_size),
        '--transformer_epochs', str(args.transformer_epochs),
        '--stack_depths', ','.join(map(str, args.stack_depths))
    ])

class Args:
    def __init__(self, model_dim=128, num_heads=4, num_layers=1, lr=1e-3, warmup=100, max_iters=1000, batch_size=64, transformer_epochs=10, stack_depths=[5, 15]):
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.lr = lr
        self.warmup = warmup
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.transformer_epochs = transformer_epochs
        self.stack_depths = stack_depths

if __name__ == '__main__':
    layers = [1, 2]
    num_heads = [2, 4]
    
    args = Args()
    run(args)