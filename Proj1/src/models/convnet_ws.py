import torch
from torch import nn


class ConvNet_ws(nn.Module):
    """
    Siamese Convolutional Network Module

    Attributes:
        conv1 (nn.Conv2d)     : fist convolutional layer
        conv2 (nn.Conv2d)     : second convolutional layer
        fc1 (nn.Linear)       : first fully connected layer
        fc2 (nn.Linear)       : second fully connected layer
        fc3 (nn.Linear)       : last fully connected layer
        classifier (nn.Linear): classifier before last activation
        drop (nn.Dropout)     : dropout function
        pool (nn.MaxPool2d)   : maxpool function
        relu (nn.Relu)        : relu activation function
        sigmoid (nn.Sigmoid)  : sigmoid activation function
    """
    
    def __init__(self):
        """Initialize Convolutional Neural Network"""
        super().__init__()
    
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # 32x(14-2)x(14-2) = 16x12x12
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 64x10x10  => pooling = 64x5x5
        
        # fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(10, 1)
        self.classifier = nn.Linear(2, 1)
        
        # regularizers
        self.drop = nn.Dropout(0.1)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        """
        Forward pass function used in the sub-network

        Args:
            x [float32]: input image with dimension 50x1x14x14 (for a batch size of 50)

        Returns:
            [float32]: non activated tensor of dimension 50x1
        """

        x = self.relu(self.conv1(x))
        x = self.drop(x)

        x = self.pool(self.relu(self.conv2(x)))
        x = self.drop(x)
        
        x = self.relu(self.fc1(x.flatten(start_dim=1)))
        x = self.drop(x)
        
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        
        x = self.fc3(x)
        
        return x

    def forward(self, x):
        """
        Forward pass function for the global siamese CNN

        Args:
            x [float32]: input images with dimension 50x2x14x14 (for a batch size of 50)

        Returns:
            [int]: predicted probability ]0,1[
        """
        input1 = x[:, 0, :, :].view(-1, 1, 14, 14)
        input2 = x[:, 1, :, :].view(-1, 1, 14, 14)
        x1 = self.forward_once(input1)
        x2 = self.forward_once(input2)
        output = torch.cat((x1, x2), 1)
        output = self.sigmoid(self.classifier(output))
        return output.squeeze(), None
