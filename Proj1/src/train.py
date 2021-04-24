import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics import evaluate_accuracy, TrainingMetrics


def train(
    model: nn.Module, train_loader: DataLoader,
    learning_rate: float = 1e-2, weight_decay: float = 1e-3,
    epochs: int = 25,
    metrics: TrainingMetrics = TrainingMetrics(),
    run: int = 1,
    verbose: int = 0
) -> None:
    """
    Train model

    Args:
        model (nn.Module): model
        train_loader (DataLoader): data loader
        learning_rate (float, optional): learning rate. Defaults to 1e-2.
        weight_decay (float, optional): weight decay for Adam. Defaults to 1e-3.
        epochs (int, optional): number of epochs. Defaults to 25.
        metrics (metrics.TrainingMetrics): metrics object to store results in
        run (int, optional): run number when performing trials
        verbose (int, optional): print info. Defaults to 0.

    Returns:
        TrainingMetrics: metrics of training run
    """

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):
        
        acc_loss = 0.
        
        model.train()
        
        for input, target, classes in train_loader:
            
            # TODO: incorporate auxiliary loss with second param
            output, _ = model(input)
            loss = criterion(output, target.float())
            
            acc_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        metrics.add_entry(epoch, acc_loss, evaluate_accuracy(model, train_loader), run)
        if verbose > 0:
            print(metrics)
