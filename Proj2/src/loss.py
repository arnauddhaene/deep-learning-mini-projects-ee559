from torch import Tensor

from module import Module


class MSELoss(Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        
        self.prediction = prediction.view(-1)
        self.target = target
        return (prediction - target).pow(2).mean()

    def backward(self):
        grad = 2 * (self.prediction - self.target) / (self.prediction.shape[0])
        return grad.unsqueeze(1)

    
class CrossEntropyLoss(Module):
  
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        
        self.softmax_input = prediction.exp() / (prediction.exp().sum(1).repeat(2, 1).t())
        self.target = target

        return - self.target.mul(self.softmax_input.log()).sum(1).mean()

    def backward(self):
        return - (self.target - self.softmax_input)
