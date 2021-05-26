import time

import torch
from torch import nn

from metrics import TrainingMetrics, TestingMetrics
from models.mlp import MLP
from models.siamese_mlp import SiameseMLP
from models.convnet import ConvNet
from models.siamese_convnet import SiameseConvNet
from train import train
from utils import load_dataset


def run(model: nn.Module, siamese: bool = False, epochs: int = 25,
        lr: float = 5e-3, decay: float = 1e-3, gamma: float = 0., trials: int = 0, seed: int = 27,
        batch_size: int = 50, standardize: bool = True,
        filename: str = 'model', verbose: int = 0):
    """
    Run a trial of model trainings.

    Args:
        model (nn.Module): Model
        siamese (bool, optional): Use the siamese weight-sharing version. Defaults to False.
        epochs (int, optional): Number of training epochs. Defaults to 25.
        lr (float, optional): Learning rate. Defaults to 5e-3.
        decay (float, optional): Regularization weight decay. Defaults to 1e-3.
        gamma (float, optional): Auxiliary loss gamma hyper-parameter. Defaults to 0..
        trials (int, optional): Number of trials to run. Defaults to 0.
        seed (int, optional): PyTorch manual seed to set. Defaults to 27.
        batch_size (int, optional): Batch size to use for training. Defaults to 50.
        standardize (bool, optional): Standardize data. Defaults to True.
        filename (str, optional): Filename to save metrics in. Defaults to 'model'.
        verbose (int, optional): Print out intermediate information. Defaults to 0.
    """
    training_metrics = TrainingMetrics()
    testing_metrics = TestingMetrics()

    for trial in range(trials):
        
        if verbose > 1:
            print(f"Creating {'standardized' if standardize else ''} "
                  f"DataLoaders with batch size {batch_size}...")
        torch.manual_seed(seed + trial)
        train_loader, test_loader = load_dataset(batch_size=batch_size, standardize=standardize)
        
        start = time.time()

        if siamese:
            model = SiameseConvNet() if model == 'ConvNet' else SiameseMLP()
        else:
            model = ConvNet() if model == 'ConvNet' else MLP()
            
        if verbose > 1:
            print(f"{model} instanciated with {model.param_count()} parameters.")
            
        train(model, train_loader,
              learning_rate=lr, weight_decay=decay, gamma=.5,
              epochs=epochs, metrics=training_metrics, run=trial,
              verbose=verbose)
        
        end = time.time()
        
        if verbose > 1:
            print("Evaluating performance on test set...")
        testing_metrics.add_entry(model, test_loader, (end - start) / epochs, verbose)
        
    training_metrics.save(filename + "_training_metrics.csv")
    testing_metrics.save(filename + "_testing_metrics.csv")


def run_all():
    
    # run("MLP", siamese=False, epochs=25, lr=5e-4, decay=1e-2, gamma=0.,
    #     trials=15, seed=27, batch_size=50, standardize=True,
    #     filename="MLP", verbose=1)
    
    # run("MLP", siamese=True, epochs=25, lr=5e-3, decay=1e-3, gamma=0.75,
    #     trials=15, seed=27, batch_size=50, standardize=True,
    #     filename="SiameseMLP", verbose=1)
    
    # run("ConvNet", siamese=False, epochs=25, lr=1e-3, decay=1e-3, gamma=0.,
    #     trials=15, seed=27, batch_size=50, standardize=True,
    #     filename="ConvNet", verbose=1)
    
    run("ConvNet", siamese=True, epochs=25, lr=5e-3, decay=1e-3, gamma=0.5,
        trials=1, seed=27, batch_size=50, standardize=True,
        filename="SiameseConvNet", verbose=2)


if __name__ == '__main__':
    run_all()
