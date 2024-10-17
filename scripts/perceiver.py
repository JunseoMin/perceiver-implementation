import torch.nn as nn
from layers import PerceiverLayer

from einops import rearrange, repeat
from einops.layers.torch import Reduce

'''
Implementation of model
'''

class Perceiver(nn.Module):
    def __init__(self, depth, weight_sharing, atten_dropout, ff_dropout, input_channel, n_cross_head, n_latent_heads, n_classes, d_latent  ):
        super().__init__()


        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, n_classes)
        ) 


    def forward(self):
        pass