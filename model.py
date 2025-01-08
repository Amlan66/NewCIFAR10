import torch
import torch.nn as nn

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Block 1: Regular Conv (RF: 3)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Block 2: Strided Conv (RF: 7)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Block 3: Dilated Conv (RF: 23)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv2d(96, 96, kernel_size=1)
        )
        
        # Block 4: Depthwise Separable Conv with stride (RF: 47)
        self.depthwise = nn.Conv2d(96, 96, kernel_size=3, padding=1, groups=96, stride=2)
        self.pointwise = nn.Conv2d(96, 192, kernel_size=1)
        self.block4_bn = nn.BatchNorm2d(192)
        self.block4_relu = nn.ReLU()
        self.block4_dropout = nn.Dropout(0.2)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final FC Layer with LogSoftmax
        self.fc = nn.Sequential(
            nn.Linear(192, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Depthwise Separable Convolution
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.block4_bn(x)
        x = self.block4_relu(x)
        x = self.block4_dropout(x)
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
