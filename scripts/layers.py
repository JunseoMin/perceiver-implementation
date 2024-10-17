'''
implementation of a layer (attention -> latent transformer)
'''
import sys,os
import torch.nn as nn
import torch

from latenttransformer import LatentTransformer

from modules import FourierFeaturePositionEncoding,Attention
from einops import rearrange

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from include.transformer.sublayer import PositionWiseFeedforward


class PerceiverLayer(nn.Module):
    def __init__(self,max_freq, n_head, d_head, d_byte_arr, d_latent, d_kv, input_type, device , 
                 num_bands = 4, atten_dropout = 0.1,ff_dropout = 0.1):
        
        super().__init__()
        self.cross_atten = Attention(n_head, d_byte_arr, d_latent, d_kv, d_kv, temp=d_head ** -0.5, dropout=atten_dropout)
        self.self_atten = LatentTransformer(n_head, d_head, d_kv, d_latent, atten_dropout)
        self.positional_encoding = FourierFeaturePositionEncoding(max_freq, d_byte_arr, input_type, device, num_bands)
        self.byte_FFN = PositionWiseFeedforward(d_latent,dropout=ff_dropout)
        self.latent_FFN = PositionWiseFeedforward(d_latent,dropout=ff_dropout)

    def forward(self, latent, byte_arr):
        #pass layers
        output = self.cross_atten(latent,byte_arr,byte_arr)
        output = self.byte_FFN(output)

        output = self.self_atten(output)
        output = self.latent_FFN(output)

        return output
