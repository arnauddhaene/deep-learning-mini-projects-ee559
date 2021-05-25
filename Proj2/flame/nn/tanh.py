from torch import Tensor
from .module import Module


class Tanh(Module):
    """
    hyperbolic tangeant activation function

    """
    def __init__(self):
        super(Tanh).__init__()

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass

        Arguments:
        input (torch.Tensor): Input tensor

        Returns:
        Activated input tensor (torch.tensor)
        """
        self.input = input
        return input.tanh()

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass: backpropagation of the gradient of the  activation function
        """
        return grad.mul(1 - self.input.tanh().pow(2))
