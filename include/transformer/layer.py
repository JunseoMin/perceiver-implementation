import torch.nn as nn
from sublayer import PositionWiseFeedforward
from sublayer import MultiheadAttention

class EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_inner, n_head, d_k, d_v, dropout = 0.1):
        super().__init__()
        self.atten = MultiheadAttention(n_head, dim_model, d_k, d_v, dropout=dropout)
        self.FFN = PositionWiseFeedforward(dim_model, dim_inner, dropout=dropout)
    
    def forward(self,enc_in, mask = None):
        output = self.atten(enc_in, mask)
        output = self.FFN(output)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, dim_model, dim_inner, n_head, d_k, d_v, dropout = 0.1):
        super().__init__()

        self.atten1 = MultiheadAttention(n_head, dim_model, d_k, d_v, dropout=dropout)
        self.atten2 = MultiheadAttention(n_head, dim_model, d_k, d_v, dropout=dropout)
        self.FFN = PositionWiseFeedforward(dim_model, dim_inner, dropout=dropout)

    def forward(self, dec_in, enc_out, dec_atten_mask = True, coder_mask = None):
        dec_out = self.atten1(dec_in, dec_in, dec_in, mask = dec_atten_mask) # masked self-attention
        dec_out = self.atten2(enc_out, enc_out, dec_out, mask = coder_mask)
        
        dec_out = self.FFN(dec_out)

        return dec_out

