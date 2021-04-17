import torch
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams['figure.figsize'] = [8.3, 5.1]

class TrainingMetrics:
    """
    Custom class for tracking and plotting training metrics.
    
    Attributes:
        metrics (dict): dictionary that stores metrics on epoch-key
        current (int): current epoch number stored for printing
        
    Usage:
        1. Define `metrics = TrainingMetrics()` before getting into epoch-loop
        2. Update metrics with `metrics.add_entry(e, l, a)`
        3. print(metrics) to display current metrics
    """
    
    def __init__(self):
        """Initialize with empty attributes"""
        self.metrics = {}
        self.current = None
    
    def __repr__(self) -> str:
        """
        Representation method

        Returns:
            str: representation
        """
        return (f"TrainingMetrics instance of size {len(self.metrics.keys())}")
    
    def __str__(self) -> str:
        """
        Print method

        Returns:
            str: to print when `print(self)` is called
        """
        metric = self.metrics[self.current]
        return (f"Epoch {metric['epoch']:02} \t"
                f"Loss {metric['loss']:07.3f} \t"
                f"Accuracy {metric['accuracy'] * 100:06.3f}")
        
    def add_entry(self, epoch: int, loss: float, accuracy: float) -> None:
        """
        Add entry to metrics

        Args:
            epoch (int): current epoch
            loss (float): loss
            accuracy (float): accuracy
        """
        self.current = epoch
        self.metrics[epoch] = \
            dict(epoch=epoch, loss=loss, accuracy=accuracy)
            
    def plot(self):
        """Plot metrics"""
        epochs = pd.DataFrame.from_dict(self.metrics, orient='index')
        epochs['epoch'] = epochs.index
        
        ax_loss = sns.lineplot(data=epochs, x="epoch", y="loss", label='loss')

        ax_acc = ax_loss.twinx()

        sns.lineplot(data=epochs, x="epoch", y="accuracy", 
                     label='accuracy', color='r', ax=ax_acc)

        plt.show()

def evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """
    Computes the classification accuracy of a model.

    Args:
        model (nn.Module): model of interest with output (prediction, auxiliary)
        loader (DataLoader): data loader with sample structure 
            that follows unpacking (input, target, classes)

    Returns:
        float: classification accuracy
    """
    
    accuracy = 0.
    counter = 0
    
    model.eval()
    
    with torch.no_grad():
            for (input, target, _) in loader:
                output, _ = model(input)
                
                accuracy += (output >= 0.5) == target
                counter += target.size(0)
                
    return (accuracy.sum() / counter).float().item()