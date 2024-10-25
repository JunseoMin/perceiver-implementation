import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from perceiver import Perceiver
from include.lamb import Lamb

import torch
import torch.nn as nn
from einops import rearrange

def test():
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

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
                    device = device,
                    n_bands = 64,
                    atten_dropout = 0., ff_dropout = 0.).to(device)

    model.load_state_dict(torch.load('path to pth'))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = torchvision.datasets.ImageNet(root='path_to_imagenet_test_dataset', split='train', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) 
            
            _, predicted = torch.max(outputs, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 

    accuracy = 100 * correct / total
    print(f'Correct percentage: {accuracy:.2f}%')

if __name__ == '__main__':
    test()