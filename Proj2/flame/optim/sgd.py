from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Parameters are obtained with model.param() with model a pytorch network
    p[0] is the parameter, p[1] is the respective gradient

    Attributes:
        model: model
        lr: learning rate
        momentum: momentum coefficient
    """
    def __init__(self, model, lr, momentum=0):
    
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.v = self.zero_velocity()

    def zero_velocity(self):
        """
        Initialize the velocity vector as zeros to match with parameter shape
        """
        v = []
        for p in self.model.parameters():
            v.append(p[0].clone().fill_(0))
        return v
    
    def step(self):
        """
        The SGD momentum update parameters with regard to Sutskever definition
        """
        for i, p in enumerate(self.model.parameters()):
            self.v[i] = self.momentum * self.v[i] + self.lr * p[1]
            p[0].sub_(self.v[i])
