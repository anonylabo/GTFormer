import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.linear = nn.Linear(args.seq_len, 1)

    def forward(self, X, _):
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination
        B, L, O, D = X.shape
        X = X.view(B, O * D, L)

        out = self.linear(X)
        out = out.view(B, 1, O, D)

        return out
