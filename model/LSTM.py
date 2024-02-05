import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(args.num_tiles, args.d_model, batch_first=True)
        self.linear = nn.Linear(args.d_model, args.d_model)

    def forward(self, X, _):
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination
        B, L, O, D = X.shape
        X = X.view(B * O, L, D)

        lstm_out, (h, c) = self.lstm(X)
        lstm_out = lstm_out.reshape(B, L, O, -1)[:, -1:]
        lstm_out_ = self.linear(lstm_out)
        out = torch.matmul(lstm_out_, lstm_out.permute(0, 1, 3, 2))

        return out
