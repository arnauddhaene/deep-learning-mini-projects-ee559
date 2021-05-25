from torch import Tensor
from .module import Module


class LossMSE(Module):
    """
        Mean Square Error (MSE) Loss
    """
    def __init__(self):
        super(LossMSE).__init__()

    def forward(self, prediction: Tensor, target: Tensor) -> float:
        """
        Forward Pass

        Arguments:
        Prediction (torch.Tensor): Predicted output of the network
        Target (torch.Tensor): Target of the network

        Returns:
        Mean Square Loss (float): mean((Prediction-Target)^2)
        """
        self.prediction = prediction.view(-1)
        self.target = target
        return (prediction - target).pow(2).mean()

    def backward(self) -> Tensor:
        """
        Back Propagation: back ward pass of the derivative of the MSE function
        """
        grad = 2 * (self.prediction - self.target) / (self.prediction.shape[0])
        return grad.unsqueeze(1)
