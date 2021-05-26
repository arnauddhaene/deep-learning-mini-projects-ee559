from torch import Tensor
from .module import Module


class LossCrossEntropy(Module):
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
        prediction = prediction.view(-1)
        self.target = target
        self.softmax_input = prediction.exp() / prediction.exp().sum()
        loss = -self.target.mul(self.softmax_input.log())
        loss += (1 - self.target).mul((1 - self.softmax_input).log())

        return loss.mean()

    def backward(self) -> Tensor:
        """
        Back Propagation: back ward pass of the derivative of the cross entropy function
        """
        
        grad = self.target.div(self.softmax_input)
        grad -= (1 - self.target).div(1 - self.softmax_input)

        return -grad.unsqueeze(1)
