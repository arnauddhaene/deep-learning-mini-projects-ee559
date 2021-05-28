from .optimizer import Optimizer
from torch import Tensor
from flame import nn


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer

    Attributes:
        model: model
        lr: learning rate
        momentum: momentum coefficient
    """
    def __init__(self, model: nn.Module, lr: float, momentum: float = 0.) -> None:
    
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.v = self.initialize_velocity()

    def initialize_velocity(self) -> Tensor:
        """
        Initialize the velocity vector as zeros according to parameters' shapes
        """
        v = []
        for p in self.model.parameters():
            v.append(p[0].clone().fill_(0))
        return v
    
    def step(self) -> None:
        """
        Performs a single optimization step, following Sutskever's momentum definition
        """
        for i, p in enumerate(self.model.parameters()):
            self.v[i] = self.momentum * self.v[i] + self.lr * p[1]
            p[0].sub_(self.v[i])
