from typing import List

from torch import Tensor


class Module(object):
    """
    Base class for all neural network modules.
    """

    def forward(self, *input) -> Tensor:
        """
        Forward pass
        """
        raise NotImplementedError

    def backward(self, *grad) -> Tensor:
        """
        Backward pass
        """
        raise NotImplementedError

    def parameters(self) -> List[List[Tensor]]:
        """
        Returns a list of parameters and their resepctive gradients
        """
        return []
    
    def __call__(self, *args):
        return self.forward(*args)
