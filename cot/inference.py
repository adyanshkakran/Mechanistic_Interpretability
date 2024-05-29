import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter

from data_loading import load_data, get_loaders, BracketDataset
from transformer_predictor import TransformerPredictor

dataset = pd.read_csv('../Data/new_new_balanced_brackets.csv')

model = TransformerPredictor.load_from_checkpoint('models/new_task_d_model=128-nhead=2-nlayers=1.ckpt')

ctoi = {c: i for i, c in enumerate('()&YNP')}
itoc = {i: c for c, i in ctoi.items()}

X = dataset['sequence'].apply(lambda x: [ctoi[c] for c in x] + [ctoi['&']]).tolist()
# X = torch.tensor(X, dtype=torch.long).to(model.device)

Y = dataset['stack_depth'].apply(lambda x: 1 if x > 0 else 0).tolist()

to_string = lambda x : ''.join([itoc[i] for i in x])

preds = []
outputs = []
BATCH_SIZE = 1
for i in tqdm(range(0, len(X), BATCH_SIZE)):
    x = torch.tensor(X[i:i+BATCH_SIZE], dtype=torch.long).to(model.device)
    y = Y[i:i+BATCH_SIZE]
    try:
        out, eos = model.model.complete_sequence(x)
    except Exception as e:
        print(e)
        print(f'Failed on {i}')
        continue
    out = out.cpu().detach().numpy().tolist()
    
    for i in range(len(out)):
        if eos[i] >= len(out[i]):
            preds.append(-1)
        elif out[i][eos[i]] == ctoi['Y']:
            preds.append(1)
        elif out[i][eos[i]] == ctoi['N']:
            preds.append(0)
        else:
            preds.append(-1)
        outputs.append(to_string(out[i][:eos[i]]) if eos[i] < len(out[i]) else to_string(out[i]))

# calculate accuracy between preds and Y
correct = 0
total = 0
for i in range(len(Y)):
    if preds[i] == Y[i]:
        correct += 1
    total += 1

print(f'Accuracy: {correct/total}')

c = Counter(preds)
print(c)

df = pd.DataFrame({'sequence': dataset['sequence'], 'label': Y, 'prediction': preds, 'output': outputs})

df.to_csv('outputs/new_testing_balanced.csv', index=False)