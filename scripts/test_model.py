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

def test():
    model = Perceiver(share_weights = True, depth = 6,
                    n_classes = 1000,
                    in_axis = 2,
                    in_channel=3,
                    max_freq = 10, 
                    n_cross_head = 1,
                    n_latent_head = 8,
                    d_cross_head = 64,
                    d_latent_head = 64,
                    d_byte_arr = 224*224,   # for fourier encoding
                    d_latent = 512,
                    n_latent = 1024,
                    d_kv = 64,
                    input_type = torch.float32,
                    device = 'gpu' if torch.cuda.is_available() else 'cpu',
                    n_bands = 64,
                    atten_dropout = 0., ff_dropout = 0.)

    model.load_state_dict(torch.load('path to pth'))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = torchvision.datasets.ImageNet(root='path_to_imagenet_test_dataset', split='train', transform=transform)

    correct = 0
    for img,label in test_dataset:
        output = model(img)

        _,predict_label = torch.max(output,1)

        if predict_label == label:
            correct += 1

    print('correct percentage: {}%', correct/len(test_dataset))


if __name__ == '__main__':
    test()