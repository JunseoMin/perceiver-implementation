import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from perceiver import Perceiver
from include.lamb import Lamb

# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        
        if i % 100 == 99:
            print(f'Batch {i+1}, Loss: {running_loss / 100:.3f}, Accuracy: {100 * correct / total:.2f}%')
            running_loss = 0.0

# Validation function
def validate_model(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss:.3f}, Accuracy: {accuracy:.2f}%')

    return val_loss, accuracy

# Parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'

epochs = 120
initial_learning_rate = 0.004

# Model Initialization
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

# Optimizer and Scheduler
optimizer = Lamb(model.parameters(), lr=initial_learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[84, 102, 114], gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ImageNet Dataset and DataLoader
train_dataset = torchvision.datasets.ImageNet(root='path_to_imagenet', split='train', transform=transform)
val_dataset = torchvision.datasets.ImageNet(root='path_to_imagenet', split='val', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

best_val_acc = 0.0

# Training loop
for epoch in range(1, epochs + 1):
    print(f'Epoch {epoch}/{epochs}')
    train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_model(model, val_loader, criterion, device)
    
    # Step the learning rate scheduler
    scheduler.step()

    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_perceiver_weights.pth")
        print(f"Best model saved at epoch {epoch}, Accuracy: {val_acc:.2f}%")

# Save the final trained weights
torch.save(model.state_dict(), "final_perceiver_weights.pth")
print("Final model weights saved.")
