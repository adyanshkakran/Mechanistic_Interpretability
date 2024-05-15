import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd

def load_from_csv(path):
    return pd.read_csv(path)

class BracketDataset(Dataset):
    def __init__(self, data=None, batch_size=64):
        self.batch_size = batch_size
        
        self.encode_dict = {'(': [1, 0, 0], ')': [0, 1, 0], '-': [0, 0, 1]}
        
        self.X = data['sequence'].values
        self.stack_depth = data['stack_depth'].values
        self.stack_depth = np.abs(self.stack_depth)
        self.Y = np.array([[0, 1] if int(y) > 0 else [1, 0] for y in self.stack_depth])
        
        self.max_len = max([len(seq) for seq in self.X])
        self.eos_index = np.array([len(seq) for seq in self.X])

        self.X_encoded = np.array([self.encode_sequence(self.pad_sequence(seq)) for seq in self.X])
        
    def encode_sequence(self, seq):
        return np.array([self.encode_dict[c] for c in seq])
    
    def pad_sequence(self, seq):
        return seq + '&' + '-' * (self.max_len - len(seq))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'x': self.X_encoded[idx], 'y': self.Y[idx], 'eos': self.eos_index[idx], 'sd': self.stack_depth[idx]}

def load_data(path):
    data = load_from_csv(path)
    # take 70% of the data for training
    # data = data.sample(frac=0.7).reset_index(drop=True)
    return BracketDataset(data)

def get_loaders(data, batch_size=32, return_data=False, train_frac=0.6):
    torch.manual_seed(0)
    
    bracket_size = len(data)
    train_size = int(train_frac * bracket_size)
    val_size = int(0.1 * bracket_size)
    test_size = bracket_size - train_size - val_size

    train_bracket, val_bracket, test_bracket = torch.utils.data.random_split(data, [train_size, val_size, test_size])

    train_loader = DataLoader(train_bracket, batch_size=batch_size, shuffle=True, num_workers=torch.get_num_threads())
    val_loader = DataLoader(val_bracket, batch_size=batch_size, num_workers=torch.get_num_threads())
    test_loader = DataLoader(test_bracket, batch_size=batch_size, num_workers=torch.get_num_threads())
    
    if return_data:
        return train_loader, val_loader, test_loader, train_bracket, val_bracket, test_bracket
    
    return train_loader, val_loader, test_loader

def remove_batch_dimension(outbeddings, stack_depths=[5, 15]):
    outbeds = torch.cat([x[0] for x in outbeddings], dim=0)
    depths = torch.cat([x[1] for x in outbeddings])

    res = torch.where(torch.isin(depths, torch.tensor(stack_depths)))
    indices = res[0]
    
    print(indices, res, len(res), indices.shape)
    

    outbeds = outbeds[indices]
    depths = depths[indices]

    return outbeds, depths

def get_probe_data(outbeddings, stack_depths=[5, 15]):
    outbeddings = [(x.cpu(), y.cpu()) for x, y in outbeddings]
    outbeds, depths = remove_batch_dimension(outbeddings, stack_depths=stack_depths)

    return outbeds, depths

def make_dataset(X,y):
    return TensorDataset(X.cpu(), y.cpu())

def get_probe_loaders(train_data, val_data, test_data, batch_size=64):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def get_stack_depths(data, stack_depths=[5, 15]):    
    stack_data = torch.where(torch.isin(data, torch.tensor(stack_depths)))[0]

    loader = DataLoader(stack_data, batch_size=64)
    
    return loader


if __name__ == '__main__':
    data = load_data('brackets.csv')
    print(data[0])