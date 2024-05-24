import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CLAAttention(nn.Module):
    def __init__(self, d_model, nhead, shared_layers):
        super(CLAAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.shared_layers = shared_layers
        self.attn = nn.MultiheadAttention(d_model, nhead)
        
        # Create shared key/value projections
        self.kv_projection = nn.Linear(d_model, d_model * 2)

    def forward(self, x, prev_kv=None):
        if prev_kv is None:
            kv = self.kv_projection(x)
            k, v = torch.chunk(kv, 2, dim=-1)
        else:
            k, v = prev_kv

        q = x
        attn_output, _ = self.attn(q, k, v)
        return attn_output, (k, v)

class CLAEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, shared_layers=2):
        super(CLAEncoderLayer, self).__init__()
        self.self_attn = CLAAttention(d_model, nhead, shared_layers)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.shared_layers = shared_layers
        self.prev_kv = None

    def forward(self, src):
        if self.prev_kv is None or self.shared_layers <= 1:
            src2, self.prev_kv = self.self_attn(src)
        else:
            src2, _ = self.self_attn(src, prev_kv=self.prev_kv)
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, nhid, nlayers, shared_layers=2, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.shared_layers = shared_layers

        encoder_layers = nn.ModuleList([CLAEncoderLayer(d_model, nhead, nhid, dropout, shared_layers) for _ in range(nlayers)])
        self.transformer_encoder = nn.Sequential(*encoder_layers)
        
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)