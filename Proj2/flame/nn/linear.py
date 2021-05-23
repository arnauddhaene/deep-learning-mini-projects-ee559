import math
from typing import List

from torch import empty, Tensor

from .module import Module


class Linear(Module):
    
    def __init__(self, inf: int, ouf: int, bias: bool = True) -> None:
        """
        Constructor

        Args:
            inf (int): input features
            ouf (int): output features
            bias (bool, optional): include bias. Defaults to True.
        """
        super(Linear).__init__()
        
        self.input_features = inf
        self.output_features = ouf
        
        self.weight = empty((self.output_features, self.input_features))
        self.loss_grad_weight = empty((self.output_features, self.input_features)).fill_(0)
        
        if bias:
            self.bias = empty(self.output_features).fill_(0)
            self.loss_grad_bias = empty(self.output_features).fill_(0)
        
    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            input (Tensor): input tensor

        Returns:
            Tensor: model prediction
        """
        self.in_ = input
        
        output = self.in_.mm(self.weight.t())
        
        if self.bias is not None:
            output += self.bias
    
        return output

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass (backprop)

        Args:
            grad (Tensor): gradient

        Returns:
            Tensor: gradient to potentially pass to next module
        """
        self.loss_grad_weight = grad.t().mm(self.in_)
        if self.loss_grad_bias is not None:
            self.loss_grad_bias = grad.t().mv(empty(grad.size(0)).fill_(1))
            
        return grad.mm(self.weight)
    
    def parameters(self) -> List[List[Tensor]]:
        """
        Fetch module parameters

        Returns:
            List[Tuple[Tensor]]: List of parameters in a List of Tuple[param, grad]
        """
        if self.bias is not None and self.loss_grad_bias is not None:
            return [[self.weight, self.loss_grad_weight], [self.bias, self.loss_grad_bias]]
        return [[self.weight, self.loss_grad_weight]]
    
    def zero_grad(self) -> None:
        """
        Zero out the gradients
        """
        self.loss_grad_weight = self.loss_grad_weight.clone().fill_(0)
        if self.loss_grad_bias is not None:
            self.loss_grad_bias = self.loss_grad_bias.clone().fill_(0)
       
    def init_weights(self) -> None:
        """
        Initialize the module's weights
        """
        self.weight = empty((self.output_features, self.input_features))\
            .normal_(0., 1 / math.sqrt(self.input_features))
