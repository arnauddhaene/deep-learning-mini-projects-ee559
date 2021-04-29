from torch import nn


class ConvNet(nn.Module):
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
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)   # 32x(14-2)x(14-2) = 32x12x12
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 64x10x10  => pooling = 64x5x5
        
        # fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10,1)
        
        # regularizers
        self.drop = nn.Dropout(0.1)
        self.drop2d = nn.Dropout2d(0.1)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass function

        Args:
            x [float32]: input images with dimension 50x2x14x14 (for a batch size of 50)

        Returns:
            [int]: predicted probability ]0,1[
        """

        x = self.drop(self.conv1(x))
        x = self.relu(x)

        x = self.drop2d(self.conv2(x))
        x = self.relu(self.pool(x))

        x = self.drop(self.fc1(x.flatten(start_dim=1)))
        x = self.relu(x)
        
        x = self.drop(self.fc2(x))
        x = self.relu(x)

        # auxiliary = x.view(-1,2,10) à tester encore

        x = self.drop(self.fc3(x))
        x = self.relu(x)

        x = self.fc4(x)
        
        x = self.sigmoid(x)
        
        return x.squeeze(), None # auxiliary
