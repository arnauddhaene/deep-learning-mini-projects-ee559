from torch import empty, Tensor
import math

from module import Module


class Linear(Module):
    
    def __init__(self, inf: int, ouf: int, bias: bool = True) -> None:
        
        super(Linear).__init__()
        
        self.input_features = inf
        self.output_features = ouf
        
        self.weight = empty((self.output_features, self.input_features))
        self.loss_grad_weight = empty((self.output_features, self.input_features)).fill_(0)
        
        if bias:
            self.bias = empty(self.output_features).fill_(0)
            self.loss_grad_bias = empty(self.output_features).fill_(0)
        
    def forward(self, input: Tensor) -> Tensor:
        
        self.in_ = input
        
        output = self.in_.mm(self.weight.t())
        
        if self.bias is not None:
            output += self.bias
    
        return output

    def backward(self, grad: Tensor) -> Tensor:
        
        self.loss_grad_weight = grad.t().mm(self.in_)
        if self.loss_grad_bias is not None:
            self.loss_grad_bias = grad.t().mv(empty(grad.size(0)).fill_(1))
            
        return grad.mm(self.weight)
    
    def parameters(self):
        if self.bias is not None and self.loss_grad_bias is not None:
            return [(self.weight, self.loss_grad_weight), (self.bias, self.loss_grad_bias)]
        return [(self.weight, self.loss_grad_weight)]
    
    def zero_grad(self) -> None:
        
        self.loss_grad_weight = self.loss_grad_weight.clone().fill_(0)
        if self.loss_grad_bias is not None:
            self.loss_grad_bias = self.loss_grad_bias.clone().fill_(0)
       
    def init_weights(self) -> None:
              
        self.weight = empty((self.output_features, self.input_features))\
            .normal_(0., 1 / math.sqrt(self.input_features))
