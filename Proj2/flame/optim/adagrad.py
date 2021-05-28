from .optimizer import Optimizer
from torch import Tensor
from flame import nn


class Adagrad(Optimizer):
    """
    Adaptative Gradient optimizer

    Attributes:
        model: model
        lr: learning rate
        eps: term added to the denominator to improve numerical stability
    """
    def __init__(self, model: nn.Module, lr: float, eps: float = 1e-8) -> None:
    
        self.model = model
        self.lr = lr
        self.eps = eps
        self.n = self.initialize_rate()

    def initialize_rate(self) -> Tensor:
        """
        Initialize the rate vector as zeros according to parameters' shapes
        """
        n = []
        for p in self.model.parameters():
            n.append(p[0].clone().fill_(0))
        return n
    
    def step(self) -> None:
        """
        Performs a single optimization step, following Adagrad algorithm
        """
        for i, p in enumerate(self.model.parameters()):
            self.n[i] = self.lr / (p[1]**2 + self.eps).sqrt()
            p[0].sub_(self.n[i] * p[1])
