import torch.nn as nn
import torch

from modules import PositionalEncoding
from layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, n_src, d_in_vec, n_layers, n_head, d_k, d_v, d_model, d_inner,
                 pad_idx, dropout, n_pos, scale_embedding = False):
        super().__init__()

        self.src_embed = nn.Embedding(n_src, d_in_vec, pad_idx)
        self.position_enc = PositionalEncoding(d_in_vec,n_pos)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([EncoderLayer(d_model,d_inner,n_head,d_k,d_v, dropout=dropout)] for _ in range(n_layers))
        self.layer_norm  = nn.LayerNorm(d_model,eps=1e-6)

        self.scale_embedding = scale_embedding
        self.d_model = d_model

    def foward(self, src_seq, src_mask):
        
        enc_output = self.src_embed(src_seq)
        if self.scale_embedding:
            enc_output *= self.d_model ** 0.5
        
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output,src_mask)
        
        return enc_output


