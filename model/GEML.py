import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.geml_layers import Grid_Embedding


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.spatLayer = Grid_Embedding(args.num_tiles, args.d_model)
        self.spat_layer_norm = nn.LayerNorm(args.d_model * 2)
        self.tempLayer = nn.LSTM(input_size=args.num_tiles, hidden_size=args.d_model, batch_first=True)
        self.temp_layer_norm = nn.LayerNorm(args.d_model)
        self.linear = nn.Linear(in_features=args.d_model, out_features=args.d_model)

        self.d_model = args.d_model

    def forward(self, X, dis_matrix):
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination
        B, L, O, D = X.shape
        spat_out = self.spatLayer(X, dis_matrix)

        temp_out, (h, _) = self.tempLayer(spat_out.view(B * O, L, D))
        temp_out = temp_out.reshape(B, L, O, -1)[:, -1:]
        temp_out_ = self.linear(temp_out)
        out = torch.matmul(temp_out_, temp_out.permute(0, 1, 3, 2))

        return out
