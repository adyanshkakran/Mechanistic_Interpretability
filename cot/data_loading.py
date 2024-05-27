import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

class BracketDataset(Dataset):
    def __init__(self, data=None):
        self.data = []
        self.ctoi = {c: i for i, c in enumerate('()&YNP')}
        
        max_len = 0
        
        for i, row in tqdm(data.iterrows(), total=len(data), desc='Generating sequences'):
            x = row['sequence']
            out = self.generate_output(x)
            self.data.append([self.ctoi[c] for c in out])
            max_len = max(max_len, len(out))
            
        for i in tqdm(range(len(self.data)), desc='Padding sequences'):
            self.data[i] += [self.ctoi['P']] * (max_len - len(self.data[i]))
            
        self.data = torch.tensor(self.data, dtype=torch.long)
        
    def step(self, seq):
        out = ""
        done = False
        seen_open = False
        for i in seq:
            if i == '(':
                out += '('
                seen_open = True
            elif i == ')':
                if not done and seen_open:
                    done = True
                    out = out[:-1]
                    continue
                out += ')'
        return out
    
    def generate_output(self, seq):
        out = seq
        while True:
            next = self.step(seq)
            if next == seq:
                out += '&N'
                break
            if next == '':
                out += '&Y'
                break
            out += ('&' if out != "" else "") + next
            seq = next
        return out
        
        
    def encode_sequence(self, seq):
        return np.array([self.input_dict[c] for c in seq])
    
    def pad_sequence(self, seq, max_len):
        return seq + (max_len - len(seq)) * '-'


    def encode_sequence(self, seq):
        return np.array([self.input_dict[c] for c in seq])
    
    def pad_sequence(self, seq, max_len):
        return seq + (max_len - len(seq)) * '-'

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, 1:]

def load_data(path):
    data = pd.read_csv(path)
    return BracketDataset(data)

def get_loaders(data, batch_size=64, return_data=False, train_frac=0.7, val_frac=0.1):
    torch.manual_seed(0)
    
    bracket_size = len(data)
    train_size = int(train_frac * bracket_size)
    val_size = int(val_frac * bracket_size)
    test_size = bracket_size - train_size - val_size

    train_bracket, val_bracket, test_bracket = torch.utils.data.random_split(data, [train_size, val_size, test_size])

    train_loader = DataLoader(train_bracket, batch_size=batch_size, shuffle=True, num_workers=torch.get_num_threads())
    val_loader = DataLoader(val_bracket, batch_size=batch_size, num_workers=torch.get_num_threads())
    test_loader = DataLoader(test_bracket, batch_size=batch_size, num_workers=torch.get_num_threads())
    
    if return_data:
        return train_loader, val_loader, test_loader, train_bracket, val_bracket, test_bracket
    
    return train_loader, val_loader, test_loader