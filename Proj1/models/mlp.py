import torch
import math
from torch import optim
from torch import Tensor
from torch import nn
import dlc_practical_prologue as prologue

## Define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2*14 * 14, 128)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(128, 20)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 2)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        #self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.flatten(start_dim=1)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x,None
