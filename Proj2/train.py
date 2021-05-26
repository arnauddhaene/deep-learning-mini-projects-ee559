from torch import set_grad_enabled, Tensor

import flame
from flame import nn

from metrics import evaluate_accuracy


def train(model: nn.Module, train_input: Tensor, train_target: Tensor,
          learning_rate: float = 1e-3, epochs: int = 50, batch_size: int = 50,
          metrics: list = [], trial: int = 0, optim: str = 'Adam', crit: str = 'MSE',
          verbose: int = 0) -> None:
    """
    Train the model

    Args:
        model (Module): model
        train_input (Tensor): input data
        train_target (Tensor): target classes
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-1.
        epochs (int, optional): Number of training epochs. Defaults to 25.
        batch_size (int, optional): Wanted batch size for training. Defaults to 1000.
        metrics (list, optional): metrics to append to. Defaults to {}.
        trial (int, optional): trial. Defaults to 0.
        optim (str, optional): optimizer. Defaults to flame.optim.SGD.
        crit (str, optional): loss criterion. Defaults to nn.LossMSE.
        verbose (int, optional): Display intermediate information. Defaults to 0.
    """
    
    set_grad_enabled(False)
    
    if crit == 'MSE':
        criterion = nn.LossMSE()
    elif crit == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise(f'Loss criterion {crit} not implemented.')
    
    if optim == 'SGD':
        optimizer = flame.optim.SGD(model, learning_rate)
    elif optim == 'Adam':
        optimizer = flame.optim.Adam(model, learning_rate)
    elif optim == 'Adagrad':
        optimizer = flame.optim.Adagrad(model, learning_rate)
    else:
        raise(f'Optimizer {optim} not implemented.')
    
    for epoch in range(epochs):
        
        acc_loss = 0.
        
        model.train()
        
        for batch in range(0, train_input.size(0), batch_size):
            
            model.zero_grad()
            
            prediction = model(train_input.narrow(0, batch, batch_size))
            
            loss = criterion(prediction.flatten(), 
                             train_target.narrow(0, batch, batch_size).float())
            
            acc_loss += loss.item()
            
            model.backward(criterion.backward())
            optimizer.step()
            
        accuracy = evaluate_accuracy(model, train_input, train_target)
        
        metrics.append({
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'epoch': epoch,
            'trial': trial,
            'optimizer': optim,
            'criterion': crit,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        })
        
        if verbose > 0 and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:02} \t"
                  f"Loss {loss:04.3f} \t"
                  f"Acc. {accuracy * 100:06.3f}")
