from torch import Tensor

from .module import Module


class ReLU(Module):
    """
    Rectified Linear Unit Activation function: max(0,x)

    """
    def __init__(self):
        super(ReLU).__init__()

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass: this function uses clamping to obtain the ReLU activation funciton

        Arguments:
        input (torch.Tensor): Input tensor

        Returns:
        Activated input tensor (torch.tensor)
        """
        self.input = input
        return input.clamp(min=0)

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass: backpropagation of the gradient of the  activation function
        """
        return grad.mul((self.input > 0).float())
