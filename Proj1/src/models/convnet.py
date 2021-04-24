from torch import nn


class ConvNet(nn.Module):
    # TODO: finish the documentation for this class @lacoupe
    """[summary]

    Attributes:
        conv1 (nn.Conv2d):
    """
    
    def __init__(self):
        """Initialize Convolutional Neural Network"""
        
        super().__init__()
        # TODO: redefine this model @lacoupe
        self.conv1 = nn.Conv2d(2, 24, kernel_size=3)
        self.conv2 = nn.Conv2d(24, 49, kernel_size=3)
        
        self.fc1 = nn.Linear(196, 128)
        self.fc2 = nn.Linear(128, 20)
        self.fc3 = nn.Linear(20, 10)
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
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        x = self.relu(self.fc1(x.flatten(start_dim=1)))
        x = self.drop(x)
        
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        
        x = self.relu(self.fc3(x.flatten(start_dim=1)))
        
        x = self.sigmoid(self.classifier(x))
        
        return x.squeeze(), None
