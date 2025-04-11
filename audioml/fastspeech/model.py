import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, n_head, dropout=0.5, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_head = n_head
        self.head_dim = self.d_out // self.n_head

        assert self.head_dim * self.n_head == self.d_out, "Head dimension is not matching with model dimension"

        # Key, Query and Value projection
        self.w_k = nn.Linear(self.d_in, self.d_out, bias=qkv_bias)
        self.w_q = nn.Linear(self.d_in, self.d_out, bias=qkv_bias)
        self.w_v = nn.Linear(self.d_in, self.d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.w_o = nn.Linear(self.d_out, self.d_out, bias=True)

    def forward(self, x, mask=None):
        batch, seq_len, emb_dim = x.shape[0], x.shape[1], x.shape[2]
        print(f"batch: {batch} | seq_len: {seq_len} | emb_dim: {emb_dim}")
        key = self.w_k(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2) # (batch, seq_len, d_in) --> (batch, n_head, seq_len, head_dim)
        query = self.w_q(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2) # (batch, seq_len, d_in) --> (batch, n_head, seq_len, head_dim)
        value = self.w_v(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2) # (batch, seq_len, d_in) --> (batch, n_head, seq_len, head_dim)
        print(f"Shape | key: {key.shape}, query: {query.shape}, value: {value.shape}")
        attn_scores = query @ key.transpose(2, 3) # --> (batch, n_head, seq_len, head_dim) * (batch, n_head, head_dim, seq_len) = (batch, n_head, seq_len, seq_len)
        # attn_scores = torch.bmm(query, key.transpose(2, 3)) 

        if mask:
            # Mask Calculation
            # Column Masking mask --> (batch x seq_len)
            seq_len = mask.shape[-1]
            assert seq_len == attn_scores.shape[-1], "Mask sequence length is NOT equal to attention scores"
            col_token_mask = mask.unsqueeze(1).expand(-1, seq_len, -1) # --> (batch x seq_len x seq_len)
            row_token_mask = mask.unsqueeze(2).expand(-1, -1, seq_len) # --> (batch x seq_len x seq_len)
            mask = col_token_mask | row_token_mask # --> (batch x seq_len x seq_len)
            mask = mask.unsqueeze(1).expand(-1, self.n_head, -1, -1) # --> (batch x n_head x seq_len x seq_len)
            attn_scores = attn_scores.masked_fill(mask, -float('inf'))# --> (batch x n_head x seq_len x seq_len)

        attn_weights = torch.softmax(
            attn_scores / key.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ value # --> (batch x n_head x seq_len x head_dim)
        # context_vec = torch.bmm(attn_weights, value).transpose(1, 2) 
        context_vec = context_vec.transpose(1, 2).contiguous().view(
            batch, seq_len, self.d_out
        )

        context_vec = self.w_o(context_vec)
        return context_vec
    

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
    

class FFTBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # MHA Layer params
        self.d_in = cfg['d_in']
        self.d_out = cfg['d_out']
        self.n_head = cfg['n_head']
        self.dropout = cfg['drop_rate']
        self.qkv_bias = cfg['qkv_bias']

        self.mha = MultiHeadAttention(
            d_in=self.d_in,
            d_out=self.d_out,
            n_head=self.n_head,
            dropout=self.dropout,
            qkv_bias=self.qkv_bias
        )
        self.layer_norm1 = LayerNorm(
            emb_dim=self.d_out
        )

        self.conv1d = Conv1D(
            emb_dim=self.d_out,
            kernel_size=cfg['kernel_size'],
            stride=cfg['stride'],
            padding=cfg['padding'],
            dilation=cfg['dilation']
        )
        self.layer_norm2 = LayerNorm(
            emb_dim=self.d_out
        )
        self.dropout = nn.Dropout(self.dropout)
        

    def forward(self, x): # --> (batch x seq_len x d_in)
        
        residual = x
        x = self.layer_norm1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.layer_norm2(x)
        x = self.conv1d(x.transpose(1, 2))
        x = self.dropout(x)
        x = x + residual
        
        return x