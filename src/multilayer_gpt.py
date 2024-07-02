import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

"""_summary_
"""
def generate_square_mask(size):
    # create a mask to prevent attention to future positions
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

"""_summary_

Returns:
    _type_: _description_
"""
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_layer, dropout, device = "cpu"):
        super(DecoderBlock, self).__init__()
        
        self.device = device
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embed_dim, ff_hidden_layer)
        self.linear2 = nn.Linear(ff_hidden_layer, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, target_mask):
        attention_output, _ = self.self_attention(x, x, x, attn_mask=target_mask)
        x = x + self.dropout1(attention_output)
        x = self.norm1(x)
        ff_output = self.linear2(F.relu(self.linear1(x)).to(self.device))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x

"""_summary_

Returns:
    _type_: _description_
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, device="cpu", max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # create division of 10,000 ^ (2i / d)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # set positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

"""_summary_

Returns:
    _type_: _description_
"""
class MultiLayerTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_layer, dropout, num_layers, device = "cpu"):
        super(MultiLayerTransformerDecoder, self).__init__()
        
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.transformer_blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_hidden_layer, dropout, device)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for transformer_block in self.transformer_blocks:
            target_mask = generate_square_mask(x.size(0)).to(self.device)
            x = transformer_block(x, target_mask)
        output = self.linear(x)
        output = self.softmax(output)
        return output

