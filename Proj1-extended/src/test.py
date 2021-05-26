import os
import click
import time
import datetime as dt

import torch

import metrics
from metrics import TrainingMetrics, TestingMetrics
from models.mlp import MLP
from models.siamese_mlp import SiameseMLP
from models.convnet import ConvNet
from models.siamese_convnet import SiameseConvNet
from train import train
from utils import load_dataset


@click.command()
@click.option('--model', default='ConvNet',
              type=click.Choice(['ConvNet', 'MLP'], case_sensitive=False),
              help="Model to evaluate.")
@click.option('--siamese/--no-siamese', default=False, type=bool,
              help="Use a siamese version of the model.")
@click.option('--epochs', default=25,
              help="Number of training epochs.")
@click.option('--lr', default=1e-2,
              help="Learning rate.")
@click.option('--decay', default=1e-3,
              help="Optimizer weight decay.")
@click.option('--gamma', default=.5,
              help="Auxiliary contribution.")
@click.option('--trials', default=1,
              help="Number of trials to run.")
@click.option('--seed', default=27,
              help="Seed for randomness.")
@click.option('--batch-size', default=50,
              help="Batch size for training.")
@click.option('--standardize/--dont-standardize', default=True, type=bool,
              help="Standardize train and test data with train data statistics.")
@click.option('--make-figs/--no-figs', default=True, type=bool,
              help="Create figures for the trial.")
@click.option('--clear-figs/--keep-figs', default=False, type=bool,
              help="Clear the figures directory of all its contents.")
@click.option('--verbose', default=1, type=int,
              help="Print out info for debugging purposes.")
def run(model, siamese, epochs,
        lr, decay, gamma, trials, seed,
        batch_size, standardize,
        make_figs, clear_figs, verbose):
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
        make_figs ([type]): [description]
        clear_figs ([type]): [description]
        verbose ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    # Clear figures directory
    if clear_figs:
        if verbose > 0:
            print("Clearing previous figures...")
        metrics.clear_figures()
    
    # Create figures subdirectory for current run
    if make_figs:
        if verbose > 0:
            print("Creating folder for trial figures...")
        timestamp = str(dt.datetime.today())
        os.makedirs(os.path.join(metrics.FIGURE_DIR, timestamp))
    
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
              learning_rate=lr, weight_decay=decay, gamma=gamma,
              epochs=epochs, metrics=training_metrics, run=trial,
              verbose=verbose)
        # model.train(False) # TEST @lacoupe
        
        end = time.time()
        
        testing_metrics.add_entry(model, test_loader, (end - start) / epochs, verbose)
        
    if make_figs:
        training_metrics.plot(timestamp)
        testing_metrics.plot(timestamp)
        
    testing_metrics.save()
        
    return training_metrics._average_accuracy(), testing_metrics._average_accuracy()


if __name__ == '__main__':
    run()
