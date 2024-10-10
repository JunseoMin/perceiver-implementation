import torch.nn as nn
from sublayer import PositionWiseFeedFoward
from sublayer import MultiheadAttention

class EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_input, dim_hidden, n_head, h, dropout = 0.1):
        super().__init__()
        self.atten = MultiheadAttention(n_head, dim_model, h, dropout=dropout)
        self.FFN = PositionWiseFeedFoward(dim_input, dim_hidden, dropout=dropout)
    
    def foward(self,enc_in, mask = None):
        output = self.atten(enc_in, mask)
        output = self.FFN(output)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, dim_model, dim_input, dim_hidden, n_head, h, dropout = 0.1):
        super().__init__()

        self.atten1 = MultiheadAttention(n_head, dim_model, h, dropout=dropout)
        self.atten2 = MultiheadAttention(n_head, dim_model, h, dropout=dropout)
        self.FFN = PositionWiseFeedFoward(dim_input, dim_hidden, dropout=dropout)

    def foward(self, dec_in, enc_out, dec_atten_mask = True, coder_mask = None):
        dec_out = self.atten1(dec_in, dec_in, dec_in, dec_atten_mask) # masked self-attention
        dec_out = self.atten2(enc_out, enc_out, dec_out, coder_mask)
        dec_out = self.FFN(dec_out)
        
        return dec_out

