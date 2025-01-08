import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CIFAR10Net
from torch.optim.lr_scheduler import OneCycleLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchsummary import summary
from tqdm import tqdm

# CIFAR10 Mean and Std
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def get_transforms():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, 
                       min_holes=1, min_height=16, min_width=16,
                       fill_value=CIFAR10_MEAN, p=0.5),
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2()
    ])
    
    test_transform = A.Compose([
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2()
    ])
    
    return train_transform, test_transform

def train(model, device, train_loader, optimizer, criterion, scheduler):
    model.train()
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({'LR': f'{current_lr:.6f}', 
                         'Train Acc': f'{100. * correct/total:.2f}%'})
    
    return 100. * correct/total

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = CIFAR10Net().to(device)
    
    # Print model summary
    summary(model, (3, 32, 32))
    
    # Dataset and DataLoader
    train_transform, test_transform = get_transforms()
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                   transform=lambda x: train_transform(image=np.array(x))["image"])
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                  transform=lambda x: test_transform(image=np.array(x))["image"])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # Training setup
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Modified scheduler parameters
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=24,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4
    )
    
    epochs = 30
    
    for epoch in range(epochs):
        train_acc = train(model, device, train_loader, optimizer, criterion, scheduler)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
    

if __name__ == '__main__':
    main()
