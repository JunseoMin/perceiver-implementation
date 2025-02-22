import torch
import torch.nn as nn

from modules import Attention

'''
Implementation of Latent Transformer

paper: deep stack of Transformer-style self-attention blocks in the latent
space.
'''

class LatentTransformer(nn.Module): ## do self 
    '''
    attention without normalization
    '''
    def __init__(self,n_head, d_head ,d_kv, d_latent, dropout = 0.1):
        super().__init__()
        self.n_head = n_head
        self.atten = Attention(n_head, d_latent, d_latent, d_kv, d_kv, temp = d_head ** -0.5, dropout = dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):

        x = self.atten(x,x,x)
        x = self.dropout(x)
        
        return x