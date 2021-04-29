import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2*14 * 14, 192) 
        self.fc2 = nn.Linear(192, 98)
        self.fc3 = nn.Linear(98, 49)
        self.fc4 = nn.Linear(49, 10)
        self.fc5 = nn.Linear(10, 1)
        # dropout layer (p=0.2)

        self.drop = nn.Dropout(0.1)
        

    def forward(self, x):
        # flatten image input
        x = x.flatten(start_dim=1) # (-1,2*14*14)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x=self.drop(x)
        x = F.relu(self.fc2(x))
        x=self.drop(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        x = x.sigmoid()
        x=x.view(-1)
        return x,None