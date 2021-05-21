import torch
from module import Module


class MSELoss(Module):

    def __init__(self):
     

    def forward(self, input, target):
    

        self.input_ = input
        self.target = target
        return (input - target).pow(2).mean()

    def backward(self):
        return 2*(self.input - self.target)/(self.input.shape[0])

    
class CrossEntropyLoss(Module):
  
    def __init__(self):
       

    def forward(self, input, target):
        
        self.softmax_input = input_.exp()/(input_.exp().sum(1).repeat(2,1).t())
        self.target = target

        return -self.target.mul(self.softmax_input.log()).sum(1).mean()

    def backward(self):
        '''
        Run the backward pass (Back-propagation), i.e. the derivative of the loss function
        '''
        return -(self.target-self.softmax_input)










