import torch.nn as nn

from layers.crowdnet_layers import STGCNBlock, TimeBlock


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.block1 = STGCNBlock(
            in_channels=args.num_tiles,
            out_channels=args.d_model,
            spatial_channels=args.d_model * 4,
            num_nodes=args.num_tiles,
        )
        self.block2 = STGCNBlock(
            in_channels=args.d_model,
            out_channels=args.d_model,
            spatial_channels=args.d_model * 4,
            num_nodes=args.num_tiles,
        )

        self.last_temporal = TimeBlock(in_channels=args.d_model, out_channels=args.num_tiles, kernel_size=3)
        self.last_conv = nn.Conv2d(in_channels=args.num_tiles, out_channels=args.num_tiles, kernel_size=(1, 1))

    def forward(self, X, A_hat):
        # X shape : (B, L, O, D)
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination
        X = X.permute(0, 2, 1, 3)

        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.last_conv(out3)

        out = out4.permute(0, 2, 1, 3)

        return out
