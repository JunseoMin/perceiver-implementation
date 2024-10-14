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
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_hid,n_position):
        super(PositionalEncoding,self).__init__()

        self.register_buffer("input_buffer",self._get_sinusoid_encoding_table(n_position=n_position,d_hid=d_hid) )

    def _get_sinusoid_encoding_table(self,n_position,d_hid):
        def get_position_angle_vec(position):
            return [position / torch.pow(10000,2 * (hid//2)) for hid in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def foward(self,x):
        return x + self.input_buffer[:, :x.size(1)].clone().detach()