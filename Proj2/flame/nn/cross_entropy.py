from torch import Tensor

from .module import Module


class CrossEntropyLoss(Module):
    """
    Cross Entropy Loss
    """
  
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass:
        Arguments:
        Prediction (torch.Tensor): Predicted output of the network
        Target (torch.Tensor): Target of the network

        Returns:
        Cross entropy loss (torch.tensor)
        """
        self.softmax_input = prediction.exp() / (prediction.exp().sum(1).repeat(2, 1).t())
        self.target = target

        return - self.target.mul(self.softmax_input.log()).sum(1).mean()

    def backward(self):
        """
        Back Propagation: back ward pass of the derivative of the cross entropy function
        """
        return - (self.target - self.softmax_input)
