from typing import Dict

from torch import set_grad_enabled, Tensor

from loss import MSELoss
from optimizer import SGD
from module import Module


def train(model: Module, train_input: Tensor, train_target: Tensor,
          learning_rate: float = 1e-1,
          epochs: int = 25, batch_size: int = 1000,
          verbose: int = 1) -> Dict:
    """
    Train model

    Args:
        model (Module): model
        train_input (Tensor): input data
        train_target (Tensor): target classes
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-1.
        epochs (int, optional): Number of training epochs. Defaults to 25.
        batch_size (int, optional): Wanted batch size for training. Defaults to 1000.
        verbose (int, optional): Display intermediate information. Defaults to 1.

    Returns:
        Dict: metrics
    """
    
    set_grad_enabled(False)
    
    criterion = MSELoss()
    
    optimizer = SGD(model, learning_rate, momentum=0.)
    
    model.train()
    
    metrics = {}
    
    for epoch in range(epochs):
        
        metrics[epoch] = {}
        
        loss = 0.
        
        for batch in range(0, train_input.size(0), batch_size):
            
            prediction = model(train_input.narrow(0, batch, batch_size))
            
            loss += criterion(prediction, train_target.narrow(0, batch, batch_size))
            
            model.zero_grad()
            model.backward(criterion.backward())
            optimizer.step()
        
        metrics[epoch]['loss'] = loss
        
        if verbose > 0:
            print(f"Epoch {epoch:02} \t"
                  f"Loss {loss:07.3f} \t")
    # TODO: add accuracy implementation
    #   f"Accuracy {accuracy * 100:06.3f}")
                
    return metrics
