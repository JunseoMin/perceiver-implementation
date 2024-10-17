import torch
import torch.nn as nn
from einops import rearrange

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from include.transformer.modules import ScaledDotProductAttention
import modules as pm

from layers import PerceiverLayer
from perceiver import Perceiver

def test_cross_attention():
    # Parameters for the CrossAttention module
    n_head = 4
    d_byte_arr = 32
    d_latent = 8
    d_k = 64
    d_v = 64
    temp = 1

    # Initialize CrossAttention module
    cross_attention = pm.Attention(n_head, d_byte_arr, d_latent, d_k, d_v, temp)

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


def test_layers():
    max_freq = 10
    n_head = 4
    d_head = 64
    d_byte_arr = 1024
    d_latent = 256
    d_kv = 64
    input_type = 'image'  
    device = 'cpu'  
    num_bands = 4
    dropout = 0.1

    model = PerceiverLayer(max_freq, n_head, d_head, d_byte_arr, d_latent, d_kv, input_type, device, num_bands, dropout)

    latent = torch.randn(8, 10, d_latent)  # (batch_size, num_latents, d_latent)
    byte_arr = torch.randn(8, 50, d_byte_arr)  # (batch_size, num_bytes, d_byte_arr)

    output = model(latent, byte_arr)

    print("Output shape:", output.shape)

    assert output.shape == latent.shape, "Output shape mismatch"
    print("Test passed!")

def test_perceiever():
    share_weights = True
    depth = 6
    n_classes = 10
    in_axis = [32, 32]
    max_freq = 10
    n_head = 4
    d_head = 64
    d_byte_arr = 2048
    d_latent = 256
    n_latent = 512
    d_kv = 64
    input_type = torch.float32
    device = 'cpu'  
    n_bands = 4
    atten_dropout = 0.1
    ff_dropout = 0.1

    
    model = Perceiver(share_weights, depth, n_classes, in_axis, max_freq, n_head, d_head, d_byte_arr, d_latent,
                      n_latent, d_kv, input_type, device, n_bands, atten_dropout, ff_dropout)

    
    batch_size = 8
    input_data = torch.randn(batch_size, *in_axis, d_byte_arr)

    
    output = model(input_data)

    
    print("Output shape:", output.shape)  # (batch_size, n_classes)

    assert output.shape == (batch_size, n_classes), f"Output shape mismatch: expected ({batch_size}, {n_classes}), but got {output.shape}"
    print("Test passed!")



if __name__ == '__main__':
    # Run the test
    test_perceiever()