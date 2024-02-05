import torch
import torch.nn as nn

from layers.gtformer_block import GTFormer_block


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.num_blocks = args.num_blocks

        self.blocks = nn.ModuleList([GTFormer_block(args) for _ in range(args.num_blocks)])

        self.out_linear = nn.Linear(args.num_tiles**2, args.num_tiles**2)
        self.relu = nn.ReLU()

    def forward(self, X, _):
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination
        B, L, O, D = X.shape

        X = torch.cat([X, torch.zeros([B, 1, O, D]).to(self.device).to(X.dtype)], dim=1).view(B, L + 1, O * D)

        if self.args.save_attention:
            A_temporals = torch.Tensor().to(X.device).to(X.dtype)

        for block in self.blocks:
            if self.args.use_only in ["temporal", "spatial"]:
                X = block(X)
            else:
                X, A_temporal, A_spatial = block(X)
                if self.args.save_attention:
                    A_temporals = torch.cat([A_temporals, A_temporal.view(B, 1, -1, L + 1, L + 1)], dim=1)

        out = self.relu(self.out_linear(X))
        out = out.view(B, L + 1, O, D)

        if self.args.save_attention:
            return out[:, -1:, :, :], A_temporals, A_spatial
        else:
            return out[:, -1:, :, :]
