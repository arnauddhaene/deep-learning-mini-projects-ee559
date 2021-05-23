from typing import Dict

from torch import set_grad_enabled, Tensor

import flame
from flame import nn
from metrics import evaluate_accuracy


def train(model: nn.Module, train_input: Tensor, train_target: Tensor,
          learning_rate: float = 5e-1,
          epochs: int = 50, batch_size: int = 25,
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
    
    criterion = nn.MSELoss()
    
    optimizer = flame.optim.SGD(model, learning_rate, momentum=0.)
    
    metrics = {}
    
    for epoch in range(epochs):
        
        model.train()
        
        metrics[epoch] = {}
        
        loss = 0.
        
        for batch in range(0, train_input.size(0), batch_size):
            
            model.zero_grad()
            
            prediction = model(train_input.narrow(0, batch, batch_size))
            
            loss += criterion(prediction, train_target.narrow(0, batch, batch_size))
            
            model.backward(criterion.backward())
            optimizer.step()
            
        accuracy = evaluate_accuracy(model, train_input, train_target)
        
        metrics[epoch]['loss'] = loss
        metrics[epoch]['accuracy'] = accuracy
        
        if verbose > 0 and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:02} \t"
                  f"Loss {loss:04.3f} \t"
                  f"Acc. {accuracy * 100:06.3f}")
                
    return metrics
