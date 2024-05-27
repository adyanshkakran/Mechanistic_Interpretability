import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd

from data_loading import load_data, get_loaders, BracketDataset
from transformer_predictor import TransformerPredictor

dataset = pd.read_csv('../Data/train-CoT-small.csv')

model = TransformerPredictor.load_from_checkpoint('models/retrain-d_model=128-nhead=8-nlayers=3.ckpt')

ctoi = {c: i for i, c in enumerate('()&YNP')}
itoc = {i: c for c, i in ctoi.items()}

X = dataset['sequence'].apply(lambda x: [ctoi[c] for c in x])
X = torch.tensor(X.tolist(), dtype=torch.long).to(model.device)
Y = dataset['stack_depth'].apply(lambda x: x > 0)
Y = torch.tensor(Y.tolist(), dtype=torch.long)

BATCH_SIZE = 4
for i in range(0, len(X), BATCH_SIZE):
    x = X[i:i+BATCH_SIZE]
    y = Y[i:i+BATCH_SIZE]
    out = model.model.complete_sequence(x)
    preds = out[:, -1].detach().cpu().numpy()
    