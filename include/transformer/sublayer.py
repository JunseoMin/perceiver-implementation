import torch.nn as nn
from modules import ScaledDotProductAttention

class MultiheadAttention(nn.Module):
    def __init__(self,n_head , d_model, d_k, d_v, dropout , mask = False):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model

        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(self.d_model, n_head * self.d_k, bias=False)
        self.w_k = nn.Linear(self.d_model, n_head * self.d_k, bias=False)
        self.w_v = nn.Linear(self.d_model, n_head * self.d_v, bias=False)

        self.w_o = nn.Linear(n_head * self.d_v, self.d_model, bias=False)    #concat weight

        self.atten = ScaledDotProductAttention()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        pass

    def forward(self,q,k,v):
        ''' update weights (forward) '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        s_batch, q_channel, k_channel, v_channel = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        #reshape
        q = self.w_q(q).view(s_batch, q_channel, n_head, d_k)
        k = self.w_k(k).view(s_batch, k_channel, n_head, d_k)
        v = self.w_v(v).view(s_batch, v_channel, n_head, d_v)

        q,k,v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)    # transpose for dot product

        q = self.atten(q,k,v)

        q = q.transpose(1,2).contiguous().view(s_batch, q_channel, -1)

        q += residual
        q = self.layer_norm(q)
        q = self.dropout(q)

        return q
    
class PositionWiseFeedforward(nn.Module):
    def __init__(self,d_in, d_hidden, dropout = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_in,d_hidden, bias=True)
        self.w2 = nn.Linear(d_hidden,d_in, bias=True)

        self.layer_norm = nn.LayerNorm()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w2(nn.functional.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual

        self.layer_norm(x)

        return x