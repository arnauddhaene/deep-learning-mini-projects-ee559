import torch.nn as nn

from models.custom import SizeableModule, NamedModule


class MLP(SizeableModule, NamedModule):
    # TODO: @pisa documentation and typing of this file
    """[summary]

    Attributes:
        fc1 ([type]): [description]
    """
    
    def __init__(self):
        """[summary]
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 98)
        self.fc3 = nn.Linear(98, 49)
        self.fc4 = nn.Linear(49, 10)
        
        self.classifier = nn.Linear(10, 1)
        
        # dropout layer
        self.drop = nn.Dropout(0.2)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        # flatten image input
        x = x.flatten(start_dim=1)  # (-1, 2x14x14)
        # add hidden layer, with relu activation function
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        
        x = self.relu(self.fc3(x))
        x = self.drop(x)
        
        x = self.fc4(x)
        x = self.sigmoid(self.classifier(x))
        
        return x.squeeze(), None
    
    def __str__(self) -> str:
        """Representation"""
        return "Multi-Layer Perceptron"
