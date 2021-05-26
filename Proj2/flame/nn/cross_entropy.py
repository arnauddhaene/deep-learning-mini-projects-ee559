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
        loss += (1 - self.target).mul((1 - self.softmax_intput).log()).mean()

        return loss

    def backward(self) -> Tensor:
        """
        Back Propagation: back ward pass of the derivative of the cross entropy function
        """
        
        grad = -self.softmax_input.div(self.target)
        grad += (1 - self.softmax_input).div(1 - self.target)

        return grad.unsqueeze(1)
