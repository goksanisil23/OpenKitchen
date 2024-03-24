import math

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=7):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)  # Note the added dimension for batch size
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x is expected to have shape [sequence_length, batch_size, feature_size]
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size=2,
        d_model=64,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers, num_layers=num_encoder_layers
        )
        self.encoder = nn.Linear(input_size, d_model)
        self.decoder = nn.Linear(d_model, 2)
        self.d_model = d_model
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src.transpose(
            0, 1
        )  # Change shape from [batch_size, seq_len, features] to [seq_len, batch_size, features]
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[-1])
        return output
