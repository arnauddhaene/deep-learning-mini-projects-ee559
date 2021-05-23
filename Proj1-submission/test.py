import time

import torch

from metrics import TrainingMetrics, TestingMetrics
from models.mlp import MLP
from models.siamese_mlp import SiameseMLP
from models.convnet import ConvNet
from models.siamese_convnet import SiameseConvNet
from train import train
from utils import load_dataset


def run(model, siamese, epochs,
        lr, decay, gamma, trials, seed,
        batch_size, standardize,
        filename, verbose):
    """[summary]

    Args:
        model ([type]): [description]
        siamese ([type]): [description]
        epochs ([type]): [description]
        lr ([type]): [description]
        decay ([type]): [description]
        gamma ([type]): [description]
        trials ([type]): [description]
        batch_size ([type]): [description]
        standardize ([type]): [description]
        filename (str): File to save results in
        verbose ([type]): [description]
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
            
        # model.train(True) # TEST @lacoupe
        train(model, train_loader,
              learning_rate=lr, weight_decay=decay, gamma=.5,
              epochs=epochs, metrics=training_metrics, run=trial,
              verbose=verbose)
        # model.train(False) # TEST @lacoupe
        
        end = time.time()
        
        testing_metrics.add_entry(model, test_loader, (end - start) / epochs, verbose)
        
    training_metrics.save(filename + "_training_metrics.csv")
    testing_metrics.save(filename + "_testing_metrics.csv")


def run_all():
    
    run("MLP", siamese=False, epochs=25, lr=5e-4, decay=1e-2, gamma=0.,
        trials=15, seed=27, batch_size=50, standardize=True,
        filename="MLP", verbose=1)
    
    run("MLP", siamese=True, epochs=25, lr=5e-3, decay=1e-3, gamma=0.75,
        trials=15, seed=27, batch_size=50, standardize=True,
        filename="SiameseMLP", verbose=1)
    
    run("ConvNet", siamese=False, epochs=25, lr=1e-3, decay=1e-3, gamma=0.,
        trials=15, seed=27, batch_size=50, standardize=True,
        filename="ConvNet", verbose=1)
    
    run("ConvNet", siamese=True, epochs=25, lr=5e-3, decay=1e-3, gamma=0.5,
        trials=15, seed=27, batch_size=50, standardize=True,
        filename="SiameseConvNet", verbose=1)


if __name__ == '__main__':
    run_all()
