import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_loading import load_data, get_loaders, BracketDataset
from transformer_predictor import TransformerPredictor

def get_name_from_config(config):
    """
    Convert a config dict to the string under which the corresponding
    models and datasets will be saved.
    """
    return f'd_model={config["model_dim"]}-nhead={config["num_heads"]}-nlayers={config["num_layers"]}'

dataset = load_data('../Data/train-CoT.csv')

BATCH_SIZE = 64
train_loader, val_loader, test_loader = get_loaders(dataset, batch_size=BATCH_SIZE)

config = {
    'model_dim': 256,
    'num_heads': 8,
    'num_layers': 3,
    'lr': 1e-3
}

model = TransformerPredictor(
    input_dim=6,
    model_dim=config['model_dim'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    lr=config['lr'],
)
name = get_name_from_config(config)
early_stopping =  EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, dirpath='models/', filename=name)
trainer = L.Trainer(max_epochs=100, devices=1, callbacks=[early_stopping, model_checkpoint])

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)

