import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter

from data_loading import load_data, get_loaders, BracketDataset
from transformer_predictor import TransformerPredictor

# dataset = pd.read_csv('../Data/count_min-0_max-0_per-count-5000_only-unbalanced_testing.csv')
dataset = pd.read_csv('../Data/test-CoT.csv')

model = TransformerPredictor.load_from_checkpoint('models/d_model=128-nhead=8-nlayers=3.ckpt')

ctoi = {c: i for i, c in enumerate('()&:#0123456789YNP')}
itoc = {i: c for c, i in ctoi.items()}

X = dataset['sequence'].apply(lambda x: [ctoi[c] for c in x] + [ctoi[':']] + [ctoi['0']] + [ctoi['&']])
X = torch.tensor(X, dtype=torch.long).to(model.device)

Y = dataset['stack_depth'].apply(lambda x: 1 if x > 0 else 0).tolist()

preds = []
outputs = []
BATCH_SIZE = 64
    
to_string = lambda x : ''.join([itoc[i] for i in x])

for i in tqdm(range(0, len(X), BATCH_SIZE)):
    x = X[i:i+BATCH_SIZE]
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
        outputs.append(to_string(out[i][:eos[i]+1]) if eos[i] < len(out[i]) else to_string(out[i]))
            

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

# df.to_csv('outputs/testing_unbalanced-0-count.csv', index=False)
df.to_csv('outputs/testing.csv', index=False)