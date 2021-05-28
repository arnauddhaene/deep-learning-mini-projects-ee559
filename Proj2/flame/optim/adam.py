from .optimizer import Optimizer
from typing import Tuple
from torch import Tensor
from flame import nn


class Adam(Optimizer):
    """
    Adaptative Moment optimizer

    Attributes:
        model: model
        lr: learning rate
        betas: hyper-parameter used for computing moving averages of gradient and its square
        eps: term added to the denominator to improve numerical stability
    """
    def __init__(self, model: nn.Module, lr: float,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
    
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = self.initialize()
        self.v = self.initialize()
        
    def initialize(self) -> Tensor:
        """
        Initialize the moving average vectors as zeros according to parameters' shapes
        """
        x = []
        for p in self.model.parameters():
            x.append(p[0].clone().fill_(0))
        return x
    
    def step(self) -> None:
        """
        Performs a single optimization step, following ADAM algorithm
        """
        for i, p in enumerate(self.model.parameters()):
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * p[1]
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * p[1]**2
            m_hat = self.m[i] / (1 - self.betas[0])
            v_hat = self.v[i] / (1 - self.betas[1])
            p[0].sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))
