from torch import nn


class ConvNet(nn.Module):
    # TODO: finish the documentation for this class @lacoupe
    """[summary]

    Attributes:
        conv1 (nn.Conv2d):
    """
    
    def __init__(self):
        """Initialize Convolutional Neural Network"""
        # input : 2x14x14
        super().__init__()
    
        self.conv1 = nn.Conv2d(2,  16, kernel_size=3) #16x(14-2)x(14-2) = 16x12x12
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3) #32x(12-2)x(12-2) = 32x10x10  => pooling = 32x5x5
        
        self.fc1 = nn.Linear(32*5*5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.classifier = nn.Linear(10, 1)
        
        self.drop = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        x = self.relu(self.conv1(x))

        x = self.pool(self.relu(self.conv2(x))) 
        
        x = self.relu(self.fc1(x.flatten(start_dim=1)))
        x = self.drop(x)
        
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        
        x = self.sigmoid(self.classifier(x))
        
        return x.squeeze(), None
