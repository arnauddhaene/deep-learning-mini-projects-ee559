class Optimizer(object):
    """
    Optimization base class
    Once the gradient is calculated through back propagation, the parameters are updated.
    """
    def step(self, *args) -> None:
        """
        Gradient Descent parameter update
        """
        raise NotImplementedError
