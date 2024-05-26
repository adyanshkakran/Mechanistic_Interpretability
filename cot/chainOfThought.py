# %%
import torch
import pytorch_lightning as L

# %%
import sys
try:
    del sys.modules['data_loading']
    del sys.modules['transformer_predictor']
except:
    pass

from data_loading import load_data, get_loaders, get_probe_data, get_probe_loaders, make_dataset
from transformer_predictor import TransformerPredictor, train, test

# %%
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

# %%
dataset = load_data('../Data/train-CoT-small.csv')

# %%
dataset[0]['x'].shape, dataset[0]['y'].shape, dataset[0]['mask'].shape

# %%
BATCH_SIZE = 64
train_loader, val_loader, test_loader, train_data, val_data, test_data = get_loaders(dataset, batch_size=BATCH_SIZE, return_data=True)

# %%
print(len(train_data), len(val_data), len(test_data), len(dataset))

# %%
model = TransformerPredictor(
    input_dim=3,
    model_dim=128,
    num_classes=5,
    num_heads=2,
    num_layers=1,
    lr=1e-3,
    warmup=50,
    max_iters=1000,
)
trainer = L.Trainer(max_epochs=300, devices=-1,accelerator="gpu", num_nodes=1, strategy='ddp')

# %%
res, _, _ = train(model, trainer, train_loader, val_loader)

# %%
res = test(model, trainer, test_loader)

# %%



