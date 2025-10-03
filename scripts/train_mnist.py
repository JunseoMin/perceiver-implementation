import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from perceiver import Perceiver
from einops import rearrange

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = rearrange(inputs, 'b c h w -> b h w c')

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f'Train Loss: {epoch_loss:.3f}, Accuracy: {epoch_acc:.2f}%')
    return epoch_loss, epoch_acc


def validate_model(model, val_loader, criterion, device):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = rearrange(inputs, 'b c h w -> b h w c')

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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 20
learning_rate = 0.001
batch_size = 64

model = Perceiver(
    share_weights=True,
    depth=4,
    n_classes=10,     # MNIST → 10 classes
    in_axis=2,
    in_channel=1,     # MNIST → 1 channel (grayscale)
    max_freq=6,
    n_cross_head=1,
    n_latent_head=4,
    d_cross_head=32,
    d_latent_head=32,
    d_byte_arr=28*28,  # MNIST image size for Fourier encoding
    d_latent=128,
    n_latent=128,
    d_kv=32,
    input_type=torch.float32,
    device=device,
    n_bands=32,
    atten_dropout=0.0,
    ff_dropout=0.0
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
val_dataset   = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


best_val_acc = 0.0
for epoch in range(1, epochs+1):
    print(f"\nEpoch {epoch}/{epochs}")
    train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_model(model, val_loader, criterion, device)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_mnist_perceiver.pth")
        print(f"Best model saved at epoch {epoch}, Accuracy: {val_acc:.2f}%")

# Save final model
torch.save(model.state_dict(), "final_mnist_perceiver.pth")
print("Training finished, final model saved!")
