import torch
import torch.nn as nn

class Embedding(nn.Module):

    def __init__(self, emb_dim, vocab_size, token_context_length):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.context_length = token_context_length

        self.tok_emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_emb = nn.Embedding(self.context_length, self.emb_dim)

    def forward(self, x):
        batch, seq_len = x.shape
        emb = self.tok_emb(x)

        pos_embeds = self.pos_emb(
            torch.arange(seq_len) # Add this tensor to device with argument device
        )
        embedding = emb + pos_embeds
        return embedding
    

class Conv1D(nn.Module):

    def __init__(self, emb_dim, kernel_size=3, stride=1, padding=None, dilation=1):
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=4*emb_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(in_channels=4*emb_dim, out_channels=emb_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
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