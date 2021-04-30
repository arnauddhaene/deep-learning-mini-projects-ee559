import torch
from torch import nn

from models.custom import SizeableModule, NamedModule


class ConvNet(SizeableModule, NamedModule):
    """
    Convolutional Network Module

    Attributes:
        conv1 (nn.Conv2d)     : fist convolutional layer
        conv2 (nn.Conv2d)     : second convolutional layer
        fc1 (nn.Linear)       : first fully connected layer
        fc2 (nn.Linear)       : second fully connected layer
        fc3 (nn.Linear)       : third fully connected layer
        fc4 (nn.Linear)       : last fully connected layer
        drop (nn.Dropout)     : dropout function
        drop2d (nn.Dropout)   : dropout function that drop entires channels
        pool (nn.MaxPool2d)   : maxpool function
        relu (nn.Relu)        : relu activation function
        sigmoid (nn.Sigmoid)  : sigmoid activation function
    """
    
    def __init__(self):
        """Initialize Convolutional Neural Network"""
        super().__init__()
    
        # convolutional layers
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3)   # 16x(14-2)x(14-2) = 16x12x12
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)  # 32x10x10  => pooling = 32x5x5
        
        # fully connected layers
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 1)
        
        # regularizers
        self.drop = nn.Dropout(0.1)
        self.drop2d = nn.Dropout2d(0.1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bn2d = nn.BatchNorm2d(16, affine=False)
        self.bn = nn.BatchNorm1d(64, affine=False)

        # activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass function

        Args:
            x [float32]: input images with dimension 50x2x14x14 (for a batch size of 50)

        Returns:
            [int]: predicted probability ]0,1[
        """

        x = self.drop(self.bn2d(self.conv1(x)))
        x = self.relu(x)

        x = self.drop2d(self.conv2(x))
        x = self.relu(self.pool(x))

        x = self.bn(self.drop(self.fc1(x.flatten(start_dim=1))))
        x = self.relu(x)
        
        x = self.drop(self.fc2(x))
        x = self.relu(x)

        x = self.drop(self.fc3(x))
        x = self.relu(x)

        x = self.sigmoid(self.fc4(x))
        
        return x.squeeze(), None
    
    def __str__(self) -> str:
        """Representation"""
        return "Convolutional Neural Network"
