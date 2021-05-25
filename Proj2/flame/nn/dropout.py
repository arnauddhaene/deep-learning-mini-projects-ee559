from .module import Module
import torch


class Dropout(Module):
    """
    Dropout Module, dropout needs to switched off during testing

    Attributes:
    p (float): the dropout probability (0.5 by default)
    test (bool): should be set to False during training and True during testing
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.test = False
        self.p = p

    def forward(self, input):
        """
        Forward pass, set's random weights to zero with a probabity p
        """
        if self.test:
            return input
        
        self.drop = torch.rand(input.shape) > self.p
        return input.mul(self.drop)

    def backward(self, grad):
        """
        Back propagation, propagated the gradient towards the rest of the net
        """
        return grad.mul(self.drop)
