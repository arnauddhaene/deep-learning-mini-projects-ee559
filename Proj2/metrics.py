from torch import Tensor

from flame import nn


def evaluate_accuracy(model: nn.Module, in_: Tensor, target: Tensor) -> float:
    """
    Computes the classification accuracy of a model.
    Args:
        model (Module): model of interest with output (prediction, auxiliary)
        input (Tensor): input data
        target (Tensor): target data
    Returns:
        float: classification accuracy
    """
    
    model.train(False)
    
    output = model(in_).flatten()
    
    return ((output > .5) == target).float().mean()
