import os
import click
import datetime as dt

import metrics
from metrics import TrainingMetrics, TestMetric, TestingMetrics
from models.convnet import ConvNet
from models.siamese_convnet import SiameseConvNet
from models.mlp import MLP
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
@click.option('--trials', default=1,
              help="Number of trials to run.")
@click.option('--clear-figs/--keep-figs', default=False, type=bool,
              help="Clear the figures directory of all its contents.")
@click.option('--verbose', default=1, type=int,
              help="Print out info for debugging purposes.")
def run(model, siamese, epochs, lr, decay, trials, clear_figs, verbose):
    
    # Clear figures directory
    if clear_figs:
        metrics.clear_figures()
    # Create figures subdirectory for current run
    timestamp = str(dt.datetime.today())
    os.makedirs(os.path.join(metrics.FIGURE_DIR, timestamp))
    
    train_loader, test_loader = load_dataset()

    training_metrics = TrainingMetrics()
    testing_metrics = TestingMetrics()

    for trial in range(trials):

        if siamese:
            model = SiameseConvNet()  # if model == 'ConvNet' else SiameseMLP()
        else:
            model = ConvNet() if model == 'ConvNet' else MLP()
            
        if verbose > 1:
            print(f"{model} instanciated with {model.param_count()} parameters.")
            
        # model.train(True) # TEST @lacoupe
        train(model, train_loader,
              learning_rate=lr, weight_decay=decay, epochs=epochs,
              metrics=training_metrics, run=trial,
              verbose=verbose)
        # model.train(False) # TEST @lacoupe
        
        testing_metrics.add_entry(model, test_loader, verbose)
    
    training_metrics.plot(timestamp)
    testing_metrics.plot(timestamp)


if __name__ == '__main__':
    run()
