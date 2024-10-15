import torch
import torch.nn as nn
from einops import rearrange

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from include.transformer.modules import ScaledDotProductAttention

'''
The implementation of modules
'''
class PAttention(ScaledDotProductAttention):
    def __init__(self, temperature, dropout=0.1):
        super().__init__(temperature, dropout)
    
    def forward(self, q, k, v, mask=None):
        # q = [batch_s, seq_len, d_latent] => q = [batch_s*n_head, seq_len, d_latent / n_head] for attention calc
        # k,v = [batch_s, (k and v)_len, d_latent] => k,v = [batch_s*n_head, (k and v)_len, d_latent / n_head] for attention calc

        att = torch.einsum('b i d , b j d -> b i j', q, k) * self.temperature  # att = torch.matmul(q, k.transpose(-1,-2))

        if mask != None:
            att = att.masked_fill(mask == 0, -1e9)

        att = nn.functional.softmax(att,dim= -1)
        output = torch.einsum('b i j , b j d -> b i d', att, v)
        
        return output


class CrossAttention(nn.Module):
    def __init__(self, n_head, d_byte_arr, d_latent, d_k, d_v, temp, dropout = 0.1):
        super().__init__()
        self.n_head , self.d_byte_arr, self.d_k, self.d_v, self.d_latent = n_head, d_byte_arr, d_k, d_v, d_latent

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


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
