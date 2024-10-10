import numpy as np

import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = dropout

    def foward(self,q,k,v, mask = None):
        att = torch.matmul(q/self.temperature, k.transpose(2,3))

        if mask != None:
            att = att.masked_fill(mask == 0, -1e9)

        att = nn.functional.softmax(att,dim= -1)
        output = torch.matmul(att , v)
        
        return output
