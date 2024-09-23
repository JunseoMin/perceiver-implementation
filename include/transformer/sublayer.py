import torch.nn as nn
from modules import ScaledDotProductAttention

class MultiheadAttention(nn.Module):
    def __init__(self,n_head , d_model, h):
        super().__init__()
        self.n_head
        self.d_model

        self.d_k = d_model/h
        self.d_v = d_model/h

        self.w_o = nn.Linear(self.d_model,)

        self.atten = ScaledDotProductAttention()
        pass

    def foward(self,q,k,v):

        pass