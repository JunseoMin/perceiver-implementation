import numpy as np

import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def foward(self,q,k,v):
        output = nn.functional.softmax(torch.matmul(q,k.transpose()) / torch.sqrt(k.dim()))
        output = torch.matmul(output , v)
        return output
