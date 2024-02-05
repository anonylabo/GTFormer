import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embed import TokenEmbedding_spatial, TokenEmbedding_temporal
from layers.self_attention import (
    AFTKVR,
    AFTFull,
    AFTSimple,
    KVR_Spatial_SelfAttention,
    Relative_Temporal_SelfAttention,
    Spatial_SelfAttention,
    Temporal_SelfAttention,
)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, x):
        new_x, A = self.attention(x)
        x = x + self.dropout1(new_x)

        y = x = self.norm1(x)
        y = self.dropout2(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.conv2(y).transpose(-1, 1)

        return self.norm2(x + y), A


class Encoder(nn.Module):
    def __init__(self, enc_layer, norm_layer):
        super(Encoder, self).__init__()
        self.enc_layer = enc_layer
        self.norm = norm_layer

    def forward(self, x):
        x, A = self.enc_layer(x)
        x = self.norm(x)

        return x, A


class GTFormer_block(nn.Module):
    def __init__(self, args):
        super(GTFormer_block, self).__init__()

        self.args = args

        if args.use_only != "spatial":
            # Temporal Transformer Block
            self.temporal_embedding = TokenEmbedding_temporal(args.num_tiles**2, args.d_model)

            if args.temporal_mode == "BRPE":
                temporal_selfattention = Relative_Temporal_SelfAttention(
                    args.d_model, args.n_head, args.seq_len + 1, args.save_attention
                )
            else:
                temporal_selfattention = Temporal_SelfAttention(args.d_model, args.n_head, args.save_attention)

            temporal_encoder_layer = EncoderLayer(
                attention=temporal_selfattention, d_model=args.d_model, d_ff=args.d_model * 4, dropout=args.dropout
            )

            temporal_norm = nn.LayerNorm(args.d_model)
            self.temporal_transformer_encoder = Encoder(temporal_encoder_layer, temporal_norm)
            self.temporal_linear = nn.Linear(args.d_model, args.num_tiles**2)

        if args.use_only != "temporal":
            # Geospatial Transformer Block
            self.spatial_embedding = TokenEmbedding_spatial(args.seq_len + 1, args.d_model)

            if args.spatial_mode == "AFT-full":
                spatial_selfattention = AFTFull(args.num_tiles, args.d_model, args.n_head, args.save_attention)
            elif args.spatial_mode == "AFT-simple":
                spatial_selfattention = AFTSimple(args.num_tiles, args.d_model, args.n_head, args.save_attention)
            elif args.spatial_mode == "KVR":
                spatial_selfattention = KVR_Spatial_SelfAttention(
                    args.num_tiles, args.d_model, args.n_head, args.save_attention
                )
            else:
                spatial_selfattention = Spatial_SelfAttention(args.d_model, args.n_head, args.save_attention)

            spatial_encoder_layer = EncoderLayer(
                attention=spatial_selfattention, d_model=args.d_model, d_ff=args.d_model * 4, dropout=args.dropout
            )

            spatial_norm = nn.LayerNorm(args.d_model)
            self.spatial_transformer_encoder = Encoder(spatial_encoder_layer, spatial_norm)
            self.spatial_linear = nn.Linear(args.d_model, args.seq_len + 1)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, X):
        if self.args.use_only == "temporal":
            temp_in = self.temporal_embedding(X)
            temp_out, A_temporal = self.temporal_transformer_encoder(temp_in)
            out = self.temporal_linear(temp_out)

            return self.dropout(out)

        elif self.args.use_only == "spatial":
            spat_in = self.spatial_embedding(X.permute(0, 2, 1))
            spat_out, A_spatial = self.spatial_transformer_encoder(spat_in)
            out = self.spatial_linear(spat_out).permute(0, 2, 1)

            return self.dropout(out)

        else:
            temp_in = self.temporal_embedding(X)
            temp_out, A_temporal = self.temporal_transformer_encoder(temp_in)
            temp_out = self.temporal_linear(temp_out)

            spat_in = self.spatial_embedding(X.permute(0, 2, 1))
            spat_out, A_spatial = self.spatial_transformer_encoder(spat_in)
            spat_out = self.spatial_linear(spat_out)

            out = temp_out + spat_out.permute(0, 2, 1)

            return self.dropout(out), A_temporal, A_spatial
