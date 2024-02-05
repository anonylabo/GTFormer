import torch
import torch.nn as nn


class Grid_Embedding(nn.Module):
    def __init__(self, num_tiles, d_model):
        super(Grid_Embedding, self).__init__()
        self.linear = nn.Linear(num_tiles * 2, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.linear3 = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model * 2, num_tiles)
        self.sigmoid = nn.Sigmoid()

        self.d_model = d_model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, X, dis_matrix):
        B, L, O, D = X.shape
        sem_neibor = X > 0
        geo_neibor = dis_matrix <= 5

        X_ = torch.cat((X, X.permute(0, 1, 3, 2)), dim=-1)
        Y = self.linear(X_)

        sqrt_dis_matrix = torch.sqrt(dis_matrix)
        sqrt_dis_matrix[~geo_neibor] = 0.0
        dis_w = sqrt_dis_matrix / torch.sum(sqrt_dis_matrix, dim=-1)[:, None]
        f = torch.matmul(dis_w, Y)
        geo_out = self.linear2(Y + f)
        geo_out = self.sigmoid(geo_out)

        tile_deg = X_.sum(-1)
        tile_deg = tile_deg[:, :, None, :].repeat(1, 1, O, 1)
        tile_deg[~sem_neibor] = 0.0
        deg_w = tile_deg / (tile_deg.sum(-1, keepdim=True) + 1e-5)
        f = torch.matmul(deg_w, Y)
        sem_out = self.linear3(Y + f)
        sem_out = self.sigmoid(sem_out)

        v = torch.cat((geo_out, sem_out), dim=-1)
        v = self.out_linear(v)

        return v
