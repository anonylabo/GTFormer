import math

import torch
import torch.nn as nn


class Relative_Temporal_SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, len_, save_attention):
        super(Relative_Temporal_SelfAttention, self).__init__()

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)

        # Relative Postion Embedding E
        self.e_projection = nn.Linear(d_model // n_head, len_, bias=False)

        self.n_head = n_head
        self.save_attention = save_attention

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_head

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, L, H, -1)
        values = self.value_projection(x).view(B, L, H, -1)

        # QE
        qe = self.e_projection(queries).permute(0, 2, 1, 3)

        # Compute S^rel
        m = nn.ReflectionPad2d((0, L - 1, 0, 0))
        qe = nn.functional.pad(m(qe), (0, 1, 0, L - 1))
        qe = qe.reshape(B, H, qe.shape[-1], qe.shape[-2])
        s_rel = qe[:, :, :L, L - 1 :]

        scale = 1.0 / math.sqrt(queries.shape[-1])

        scores = torch.einsum("blhd,bshd->bhls", queries, keys)

        A = torch.softmax(scale * (scores * s_rel), dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        out = V.contiguous().view(B, L, -1)

        if self.save_attention:
            return out, A
        else:
            return out, None


class Temporal_SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, save_attention):
        super(Temporal_SelfAttention, self).__init__()

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)

        self.n_head = n_head
        self.save_attention = save_attention

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_head

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, L, H, -1)
        values = self.value_projection(x).view(B, L, H, -1)

        scale = 1.0 / math.sqrt(queries.shape[-1])

        scores = torch.einsum("blhd,bshd->bhls", queries, keys)

        A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        out = V.contiguous().view(B, L, -1)

        if self.save_attention:
            return out, A
        else:
            return out, None


class Spatial_SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, save_attention):
        super(Spatial_SelfAttention, self).__init__()

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.n_head = n_head
        self.save_attention = save_attention

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_head

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, L, H, -1)
        values = self.value_projection(x).view(B, L, H, -1)

        scale = 1.0 / math.sqrt(queries.shape[-1])

        scores = torch.einsum("blhd,bshd->bhls", queries, keys)

        A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        out = V.contiguous().view(B, L, -1)

        if self.save_attention:
            return out, A
        else:
            return out, None


class KVR_Spatial_SelfAttention(nn.Module):
    def __init__(self, num_tiles, d_model, n_head, save_attention):
        super(KVR_Spatial_SelfAttention, self).__init__()

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.n_head = n_head
        self.save_attention = save_attention

        key_indices = []
        for i in range(num_tiles**2):
            index = []
            start = i // num_tiles
            end = i % num_tiles
            for j in range(num_tiles):
                index.append(start * num_tiles + j)
                index.append(end + num_tiles * j)
            index.remove(i)
            key_indices.append(sorted(index))
        self.key_indices = torch.tensor(key_indices)

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_head

        queries = self.query_projection(x).view(B, L, H, -1).permute(0, 2, 1, 3)
        keys = self.key_projection(x).view(B, L, H, -1).permute(0, 2, 1, 3)
        values = self.value_projection(x).view(B, L, H, -1).permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(queries.shape[-1])

        keys_sample = keys[:, :, self.key_indices, :]
        values_sample = values[:, :, self.key_indices, :]

        scores = torch.einsum("bhlkd,bhlds->bhlks", queries.unsqueeze(-2), keys_sample.transpose(-2, -1))

        A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhlks,bhlsd->bhlkd", A, values_sample).squeeze(dim=3)
        out = V.permute(0, 2, 1, 3).contiguous().view(B, L, -1)

        if self.save_attention:
            A = A.squeeze()
            A_ = torch.zeros((B, H, L, L)).to(self.device)
            for j in range(L):
                A_[:, :, j, self.key_indices[j]] = A[:, :, j, :]
            return out, A_
        else:
            return out, None


class AFTKVR(nn.Module):
    def __init__(self, num_tiles, d_model, n_head, save_attention):
        super().__init__()

        self.n_head = n_head
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.wbias = nn.Parameter(torch.Tensor(num_tiles**2, 2 * num_tiles - 1))
        self.save_attention = save_attention

        key_indices = []
        for i in range(num_tiles**2):
            index = []
            start = i // num_tiles
            end = i % num_tiles
            for j in range(num_tiles):
                index.append(start * num_tiles + j)
                index.append(end + num_tiles * j)
            index.remove(i)
            key_indices.append(sorted(index))
        self.key_indices = torch.tensor(key_indices)

        nn.init.xavier_uniform_(self.wbias)

    def forward(self, x):
        B, T, _ = x.shape
        H = self.n_head

        queries = self.query_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)
        keys = self.key_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)
        values = self.value_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)
        temp_wbias = self.wbias.unsqueeze(0).unsqueeze(1)

        queries_sig = torch.sigmoid(queries)
        kv = torch.mul(torch.exp(keys), values)[:, :, self.key_indices, :]
        temp = torch.exp(temp_wbias.unsqueeze(-2)) @ kv
        weighted = (
            temp.squeeze()
            / (torch.exp(temp_wbias.unsqueeze(-2)) @ torch.exp(keys[:, :, self.key_indices, :])).squeeze()
        )
        out = torch.mul(queries_sig, weighted)

        out = out.permute(0, 2, 1, 3).view(B, T, -1)

        if self.save_attention:
            return out, temp_wbias
        else:
            return out, None


class AFTFull(nn.Module):
    def __init__(self, num_tiles, d_model, n_head, save_attention):
        super().__init__()

        self.n_head = n_head
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.wbias = nn.Parameter(torch.Tensor(num_tiles**2, num_tiles**2))
        self.save_attention = save_attention
        nn.init.xavier_uniform_(self.wbias)

    def forward(self, x):
        B, T, _ = x.shape
        H = self.n_head

        queries = self.query_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)
        keys = self.key_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)
        values = self.value_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)

        queries_sig = torch.sigmoid(queries)
        temp = torch.exp(self.wbias[None, None, :, :]) @ torch.mul(torch.exp(keys), values)
        weighted = temp / (torch.exp(self.wbias[None, None, :, :]) @ torch.exp(keys))
        out = torch.mul(queries_sig, weighted).permute(0, 2, 1, 3).view(B, T, -1)

        if self.save_attention:
            return out, self.wbias
        else:
            return out, None


class AFTSimple(nn.Module):
    def __init__(self, num_tiles, d_model, n_head, save_attention):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)

        self.save_attention = save_attention

    def forward(self, x):
        B, T, _ = x.shape
        H = self.n_head

        queries = self.query_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)
        keys = self.key_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)
        values = self.value_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)

        weights = torch.mul(torch.softmax(keys, 1), values).sum(dim=1, keepdim=True)
        queries_sig = torch.sigmoid(queries)
        out = torch.mul(queries_sig, weights)

        out = out.permute(0, 2, 1, 3).view(B, T, -1)

        return out, None
