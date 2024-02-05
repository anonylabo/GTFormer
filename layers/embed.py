import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)].to(x.dtype)


class TokenEmbedding_temporal(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding_temporal, self).__init__()
        self.token_embedding = nn.Linear(c_in, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)

    def forward(self, x):
        x = self.token_embedding(x) + self.positional_encoding(x)
        return x


class TokenEmbedding_spatial(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding_spatial, self).__init__()
        self.token_embedding = nn.Linear(c_in, d_model)

    def forward(self, x):
        x = self.token_embedding(x)
        return x
