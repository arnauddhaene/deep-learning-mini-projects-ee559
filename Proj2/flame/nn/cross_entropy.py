from torch import Tensor

from .module import Module


class CrossEntropyLoss(Module):
  
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        
        self.softmax_input = prediction.exp() / (prediction.exp().sum(1).repeat(2, 1).t())
        self.target = target

        return - self.target.mul(self.softmax_input.log()).sum(1).mean()

    def backward(self):
        return - (self.target - self.softmax_input)
