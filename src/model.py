import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Baseline Convolutional Neural Network for CIFAR-10 classification.

    Architecture:
    - 2 Convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool)
    - Fully connected layers with Dropout for regularization.
    - Input: (3, 32, 32) tensors.
    """

    def __init__(self, n_classes: int = 10) -> None:
        super(SimpleCNN, self).__init__()

        # Block 1: Feature extraction (Low-level patterns)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: Feature extraction (Mid-level patterns)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Fully connected layers (Classification head)
        # 64 filters * (8x8) reduced spatial size after 2 poolings
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        # Conv 1 -> BN -> ReLU -> Pool (32x32 -> 16x16)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv 2 -> BN -> ReLU -> Pool (16x16 -> 8x8)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
