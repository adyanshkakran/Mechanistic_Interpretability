import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
from torch import optim
import torch.nn.functional as F
import math
import numpy as np
import torchmetrics


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerLM(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Embedding(ntoken, d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[batch_size, seq_len, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1], device=src.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

    def complete_sequence(self, src: torch.Tensor, argmax: bool =False):
        """
        Complete the sequence given by `src` with `length` elements.
        If `src` is batched, return a whole batch of predictions.
        """
        src_ = src.unsqueeze(0) if len(src.shape) == 1 else src
        done = [False] * src_.shape[0]
        output_eos = [0] * src_.shape[0]
        # src_ : [bz, seq]

        # We only work with src_ from now on. `src` is the original input.
        for i in range(3000):
            outputs = self(src_)
            # [bz, seq, ntoken]
            if not argmax:
                probabilities = F.softmax(outputs[:, -1, :], dim=-1)
                preds = torch.multinomial(probabilities, 1)
            else:
                preds = torch.argmax(outputs[:, -1:, :], dim=-1)
            # [bz, 1]
            src_ = torch.concat([src_, preds], dim=1)
            
            for i in range(src_.shape[0]):
                if not done[i] and (preds[i] == 3 or preds[i] == 4):
                    done[i] = True
                    output_eos[i] = src_.shape[1] - 1
                
            if all(done):
                break
            
        return src_, output_eos


class TransformerPredictor(L.LightningModule):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_heads,
        num_layers,
        lr,
        dropout=0.0
    ):
        """TransformerPredictor.

        Args:
            input_dim: Hidden dimensionality of the input
            model_dim: Hidden dimensionality to use inside the Transformer
            num_heads: Number of heads to use in the Multi-Head Attention blocks
            num_layers: Number of encoder blocks to use.
            lr: Learning rate in the optimizer
            dropout: Dropout to apply inside the model
        """
        super().__init__()
        self.model = TransformerLM(
            ntoken=input_dim,
            d_model=model_dim,
            nhead=num_heads,
            d_hid=model_dim,
            nlayers=num_layers,
            dropout=dropout,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None):
        return self.model(src, mask)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        src, tgt = batch
        output = self(src)
        loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        src, tgt = batch
        output = self(src)
        loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        src, tgt = batch
        output = self(src)
        loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        self.log("test_loss", loss)
        return loss
