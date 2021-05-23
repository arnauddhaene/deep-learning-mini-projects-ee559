class Optimizer(object):
    """
    Optimization base class
    The optimizer update the parameters after the gradient is calculated through back prop
    The parameters are then updated after each sample.
    """
    def step(self, *args):
        """
        Gradient Descent parameter update
        """
        raise NotImplementedError
