from torch import Tensor

from .module import Module


class Sigmoid(Module):
    """
    Sigmoid activation function

    """
    def __init__(self):
        super(Sigmoid).__init__()

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass: 

        Arguments:
        input (torch.Tensor): Input tensor

        Returns:
        Activated input tensor (torch.tensor)
        """

        self.input = input
        activated = 1 / (1 + (-input).exp())
        return activated

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass: backpropagation of the gradient of the  activation function
        """
        d_sigmoid=self.forward(self.input).mul(1 - self.forward(self.input))
        return grad.mul(d_sigmoid)
