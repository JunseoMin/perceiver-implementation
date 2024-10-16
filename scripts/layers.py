'''
implementation of a layer (attention -> latent transformer)
'''
import sys,os
import torch.nn as nn
import torch

from modules import PAttention,FourierFeaturePositionEncoding
from einops import rearrange

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from include.transformer.sublayer import PositionWiseFeedforward


class CrossAttention(nn.Module):
    def __init__(self, n_head, d_byte_arr, d_latent, d_k, d_v, temp, dropout = 0.1):
        super().__init__()
        self.n_head , self.d_byte_arr, self.d_k, self.d_v, self.d_latent = n_head, d_byte_arr, d_k, d_v, d_latent
        # dk == d_v => byte array dim

        self.w_q = nn.Linear(d_latent, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_byte_arr, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_byte_arr, n_head * d_v, bias=False)

        self.w_o = nn.Linear(n_head * d_v, d_latent, bias=False)    # attention output: latent vector

        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(d_latent, eps = 1e-6)

        self.attention = PAttention(temperature=temp, dropout = dropout)

    
    def forward(self, q, k, v, mask = None):
        '''
        each of q, k, v are Tensors
        No masks
        q => Latent array
        k,v => byte array
        '''
        residual = q

        #reshape q,k,v for dot product attention
        # q = [batch_s, seq_len, d_latent] => q = [batch_s*n_head, seq_len, d_latent / n_head] for attention calc
        # k,v = [batch_s, (k and v)_len, d_latent] => k,v = [batch_s*n_head, (k and v)_len, d_latent / n_head] for attention calc
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q,k,v = map(lambda x: rearrange(x, 'b n (h d) -> (b h) n d', h = self.n_head), (q, k, v))

        q = self.attention(q,k,v)   # scaled dot product attention

        q = rearrange(q,'(b h) n d -> b n (h d)', h=self.n_head)    # latent array

        q = self.w_o(q) #concat

        q = self.dropout(q)
        q += residual

        return q


class PerceiverLayer(nn.Module):
    def __init__(self,max_freq, n_head, d_byte_arr, d_latent, d_kv, input_type, device , num_bands = 4, temp = 0.1, dropout = 0.1):
        super().__init__()
        self.crossatten = CrossAttention(n_head, d_byte_arr, d_latent, d_kv, d_kv, temp=temp, dropout=dropout)
        self.positional_encoding = FourierFeaturePositionEncoding(max_freq, d_byte_arr, input_type, device, num_bands)
        self.FFN = PositionWiseFeedforward(d_byte_arr,d_latent,dropout=dropout)


    def forward(self, x):

        pass