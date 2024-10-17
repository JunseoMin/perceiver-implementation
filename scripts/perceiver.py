import torch
import torch.nn as nn
from layers import PerceiverLayer
from modules import FourierFeaturePositionEncoding

from einops import rearrange, repeat
from einops.layers.torch import Reduce

'''
Implementation of model
'''

def encoding_process(pos,size,device,dtype,axis):
    axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
    pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)

    return pos

class Perceiver(nn.Module):
    def __init__(self, share_weights, depth, n_classes, in_axis,
                 max_freq, n_head, d_head, d_byte_arr, d_latent, n_latent,
                 d_kv, input_type, device, n_bands, atten_dropout, ff_dropout):
        super().__init__()
        self.share_weights = share_weights
        self.depth = depth
        self.dtype = input_type
        self.device = device
        self.in_axis = in_axis

        if share_weights:
            self.layers = PerceiverLayer(n_head, d_head, d_byte_arr, d_latent,
                                         d_kv, atten_dropout, ff_dropout)
        else:
            self.layers = nn.ModuleList([PerceiverLayer(n_head, d_head, d_byte_arr, d_latent,
                                                         d_kv, atten_dropout, ff_dropout)] for _ in range(depth))

        self.latent = nn.Parameter(torch.randn(n_latent, d_latent))
        self.encoder = FourierFeaturePositionEncoding(max_freq,d_byte_arr,input_type, device, n_bands)

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, n_classes)
        ) 


    def forward(self, array, logits = True):
        #positional encoding
        b, *axis, _ = array.shape
        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=self.device, dtype=self.dtype), axis))
        pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)

        pos = self.encoder(pos)
        pos = rearrange(pos, '... n d -> ... (n d)')
        pos = repeat(pos, '... -> b ...', b = b)

        array = torch.cat((array,pos), dim=-1)
        array = rearrange(array, 'b ... d -> b (...) d')    # (batch data dim)

        latent = repeat(self.latent, 'n d -> b n d', b = b)

        if self.share_weights:
            for _ in range(self.depth):
                latent = self.layers(latent,array)
        else:
            for layer in self.layers:
                latent = layer(latent,array)

        if logits:
            return self.to_logits(latent)
        
        return latent