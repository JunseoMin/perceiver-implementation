import torch
import torch.nn as nn
from einops import rearrange

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from include.transformer.modules import ScaledDotProductAttention
import modules as pm


def test_cross_attention():
    # Parameters for the CrossAttention module
    n_head = 4
    d_byte_arr = 32
    d_latent = 8
    d_k = 64
    d_v = 64
    temp = 1

    # Initialize CrossAttention module
    cross_attention = pm.CrossAttention(n_head, d_byte_arr, d_latent, d_k, d_v, temp)

    # Sample input tensors for q, k, and v
    batch_size = 2
    seq_len_q = 10
    seq_len_kv = 15

    q = torch.randn(batch_size, seq_len_q, d_latent)   # Latent array (query)
    k = torch.randn(batch_size, seq_len_kv, d_byte_arr)  # Byte array (key)
    v = torch.randn(batch_size, seq_len_kv, d_byte_arr)  # Byte array (value)

    # Perform forward pass
    output = cross_attention(q, k, v)

    # Output the result
    print("Output shape:", output.shape)
    print("Output:", output)


if __name__ == '__main__':
    # Test the CrossAttention forward method

    # Run the test
    test_cross_attention()