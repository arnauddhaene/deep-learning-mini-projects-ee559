from torch import Tensor

from flame.nn.module import Module


class ReLU(Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
    
        self.input = input
        return input.clamp(min=0)

    def backward(self, grad: Tensor) -> Tensor:
     
        return grad.mul((self.input > 0).float())
