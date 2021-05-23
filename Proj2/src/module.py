class Module(object):
    """
    Base class for all neural network modules.
    """

    def forward(self, *input):
        """
        Forward pass
        """
        raise NotImplementedError

    def backward(self, *grad):
        """
        Backward pass
        """
        raise NotImplementedError

    def param(self):
        """
        Returns a list of parameters and their resepctive gradient
        """
        return []
    
    def __call__(self, *args):
        return self.forward(*args)
