import torch
import numpy as np
from torch.cuda import is_available
import torch.nn as nn


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Embedding(nn.Module):

    def __init__(self, emb_dim, vocab_size, token_context_length):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.context_length = token_context_length

        self.tok_emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_emb = nn.Parameter(
          get_sinusoid_encoding_table(self.context_length+1, self.emb_dim).unsqueeze(0),
          requires_grad=False,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        batch, seq_len = x.shape
        emb = self.tok_emb(x)

        pos_embeds = self.pos_emb[:, :seq_len]
        # pos_embeds = self.pos_emb(
        #     torch.arange(seq_len) # Add this tensor to device with argument device
        # )
        embedding = emb + pos_embeds
        return embedding
    

class Conv1D(nn.Module):

    def __init__(self, emb_dim, kernel_size=3, filter_size=None, stride=1, padding=None, dilation=1):
        super().__init__()

        if filter_size:
          fs_out_channels = filter_size
        else:
          fs_out_channels = 4 * emb_dim

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=fs_out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(in_channels=fs_out_channels, out_channels=emb_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
    
    def forward(self, x):
        return self.layers(x).transpose(1, 2)

class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    

class TransposedLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape)
        
    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        x = self.ln(x)
        x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
        return x