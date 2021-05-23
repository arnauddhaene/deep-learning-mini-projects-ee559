from torch import Tensor

from typing import List

from .module import Module
from .linear import Linear


class Sequential(Module):
    
    def __init__(self, modules: List[Module]) -> None:
        """
        Constructor

        Args:
            modules (List[Module]): List of modules
        """
        super(Sequential).__init__()
        self.modules = modules
        
    def forward(self, in_: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            in_ (Tensor): input tensor

        Returns:
            Tensor: model prediction
        """
        for m in self.modules:
            in_ = m.forward(in_)
        return in_
    
    def backward(self, grad: Tensor) -> None:
        """
        Backward pass (backprop)

        Args:
            grad (Tensor): criterion gradient output
        """
        for m in reversed(self.modules):
            grad = m.backward(grad)
    
    def init_weights(self) -> None:
        """
        Initialize Linear module weights
        """
        for m in self.modules:
            if isinstance(m, Linear):
                m.init_weights()
                
    def zero_grad(self):
        """
        Zero out the Linear module gradients
        """
        for m in self.modules:
            if isinstance(m, Linear):
                m.zero_grad()
                
    def parameters(self) -> List[List[Tensor]]:
        """
        Model parameters

        Returns:
            List[Tuple[Tensor]]: List of parameters in a List of Tuple[param, grad]
        """
        parameters = []
        
        for m in self.modules:
            if isinstance(m, Linear):
                parameters.extend(m.parameters())
                
        return parameters
    
    def train(self):
        pass
        # TODO: implement
        # for m in self.modules:
        #     if isinstance(m, InvertedDropout):
        #         m.eval = False

    def eval(self):
        pass
        # TODO: implement
        # for m in self.modules:
        #     if isinstance(m, InvertedDropout):
        #         m.eval = True
