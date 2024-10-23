import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from perceiver import Perceiver
from include.lamb import Lamb
import unittest

# Import the functions from the main training script
from train import train_model, validate_model

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
    
    share_weights = True
    depth = 6
    n_classes = 10
    in_axis = 2
    max_freq = 10
    n_head = 4
    d_head = 64
    d_byte_arr = 2048
    d_latent = 256
    n_latent = 512
    d_k = 64
    d_v = 64
    input_type = torch.float32
    device = 'cpu'  
    n_bands = 4
    atten_dropout = 0.1
    ff_dropout = 0.1

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
    in_axis = 2
    max_freq = 10
    
    n_cross_head = 1
    n_latent_head = 8
    d_cross_head = 64
    d_latent_head = 64

    d_byte_arr = 2048
    d_latent = 512

    n_latent = 512
    
    d_kv = 64 # input dim

    input_type = torch.float32
    device = 'cpu'  
    n_bands = 4
    atten_dropout = 0.1
    ff_dropout = 0.1

    
    model = Perceiver(share_weights, depth, n_classes, in_axis, 3, max_freq, n_cross_head,n_latent_head, d_cross_head,d_latent_head , d_byte_arr, d_latent,
                      n_latent, d_kv, input_type, device, n_bands, atten_dropout, ff_dropout)

    
    batch_size = 8

    img = torch.randn(1, 224, 224, 3) 
    
    print("INPUT shape:", img.shape)  # (batch , , byte array dim)
    # print("INPUT ", input_data)  # (batch_size, n_classes)

    output = model(img)

    
    print("Output shape:", output.shape)  # (batch_size, n_classes)

    assert output.shape == (batch_size, n_classes), f"Output shape mismatch: expected ({batch_size}, {n_classes}), but got {output.shape}"
    print("Test passed!")

'''
class TestPerceiverModel(unittest.TestCase):
    
    def setUp(self):
        """ Set up the model, optimizer, and data for testing. """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters for the Perceiver
        share_weights = True
        depth = 6
        n_classes = 1000
        in_axis = 2
        in_channel = 3
        max_freq = 10
        n_head = 64
        d_head = 64
        d_byte_arr = 50176
        d_latent = 512
        n_latent = 1024
        d_kv = 64
        input_type = torch.float32
        n_bands = 64
        atten_dropout = 0.1
        ff_dropout = 0.1
        
        # Initialize the Perceiver model
        self.model = Perceiver(share_weights, depth, n_classes, in_axis, in_channel, max_freq, n_head, d_head,
                               d_byte_arr, d_latent, n_latent, d_kv, input_type, self.device, n_bands, 
                               atten_dropout, ff_dropout).to(self.device)
        
        # Define the optimizer and loss function
        self.optimizer = Lamb(self.model.parameters(), lr=0.004, weight_decay=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()

        # Data transformation for ImageNet
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Use a smaller dataset or synthetic data for testing purposes
        # Here we will use CIFAR10 for simplicity, though your dataset may differ.
        self.train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=4, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(self.val_dataset, batch_size=4, shuffle=False, num_workers=2)

    def test_train_model(self):
        """ Test the training loop to ensure it runs without errors. """
        try:
            train_model(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
        except Exception as e:
            self.fail(f"Training failed with exception: {e}")
    
    def test_validate_model(self):
        """ Test the validation loop to ensure it runs without errors. """
        try:
            val_loss, accuracy = validate_model(self.model, self.val_loader, self.criterion, self.device)
            print(f'Validation Loss: {val_loss}, Accuracy: {accuracy}')
        except Exception as e:
            self.fail(f"Validation failed with exception: {e}")
'''
if __name__ == '__main__':
    test_perceiever()
    
    # unittest.main()
