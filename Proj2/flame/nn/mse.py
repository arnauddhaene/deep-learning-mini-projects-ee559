from torch import Tensor

from .module import Module


class MSELoss(Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction: Tensor, target: Tensor) -> float:
        
        self.prediction = prediction.view(-1)
        self.target = target
        return (prediction - target).pow(2).mean()

    def backward(self) -> Tensor:
        grad = 2 * (self.prediction - self.target) / (self.prediction.shape[0])
        return grad.unsqueeze(1)
