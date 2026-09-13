import torch.nn as nn
import torch.nn.functional as F

class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        
        self.fc1 = nn.Linear(2048, 128)   # 8 * 8 * 32 = 2048
        self.fc2 = nn.Linear(128, 10)    # 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)   # flatten everything except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
"""
Input:            3 * 32 * 32   (RGB image)

Conv1:            3 → 16 channels, kernel 3*3, padding 1, stride 1
                  → 16 * 32 * 32   (padding=1 keeps spatial size unchanged)
ReLU
Pool1:            2*2, stride 2
                  → 16 * 16 * 16   (halves spatial size)

Conv2:            16 → 32 channels, kernel 3*3, padding 1, stride 1
                  → 32 * 16 * 16   (padding keeps size unchanged again)
ReLU
Pool2:            2*2, stride 2
                  → 32 * 8 * 8   (halves again)

Flatten:          32 * 8 * 8 = 2048   (one long vector per image)

FC1:              2048 → 128
ReLU
FC2:              128 → 10   (10 CIFAR-10 classes, raw logits)
    """