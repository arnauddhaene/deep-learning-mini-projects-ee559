from torch import Tensor

from typing import List

from module import Module
from linear import Linear


class Sequential(Module):
    
    def __init__(self, modules: List[Module]) -> None:
        
        super(Sequential).__init__()
        self.modules = modules
        
    def forward(self, in_) -> Tensor:
        
        for m in self.modules:
            in_ = m.forward(in_)
        return in_
    
    def backward(self, grad: Tensor) -> Tensor:
        
        print(grad)
        
        for m in reversed(self.modules):
            grad = m.backward(grad)
        return grad
    
    def init_weights(self) -> None:
        
        for m in self.modules:
            if isinstance(m, Linear):
                m.init_weights()
                
    def zero_grad(self):
        
        for m in self.modules:
            if isinstance(m, Linear):
                m.zero_grad()
                
    def parameters(self):
        
        parameters = []
        
        for m in self.modules:
            if isinstance(m, Linear):
                parameters.extend(m.param())
                
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
