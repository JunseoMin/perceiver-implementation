import torch.nn as nn
import torch

from layer import DecoderLayer
from modules import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, n_trg, d_in_vec, n_layers, n_head, d_k, d_v ,d_model, d_inner, pad_idx, n_pos=100, dropout = 0.1, scale_emb = False):
        super().__init__()
        self.trg_emb = nn.Embedding(n_trg, d_in_vec, pad_idx)
        self.pos_encoding = PositionalEncoding(d_in_vec,n_pos)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model,d_inner,n_head,d_k,d_v,dropout=dropout)] for _ in range(n_layers))
        self.layer_norm = nn.LayerNorm(d_model,eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self,trg_seq, trg_mask, enc_output, src_mask):
        dec_output = self.trg_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        
        dec_output = self.dropout(self.pos_encoding(dec_output))
        dec_output = self.layer_norm(dec_output)

        for layer in self.layer_stack:
            dec_output = layer(dec_output, enc_output, trg_mask, src_mask)

        return dec_output

