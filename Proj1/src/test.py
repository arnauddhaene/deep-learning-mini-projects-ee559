import os
import click
import datetime as dt

import metrics
from metrics import TrainingMetrics, TestingMetrics
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
@click.option('--verbose/--no-verbose', default=False, type=bool,
              help="Print out info for debugging purposes.")
def run(model, siamese, epochs, lr, decay, trials, clear_figs, verbose):
    
    # Clear figures directory
    if clear_figs:
        metrics.clear_figures()
    # Create figures subdirectory for current run
    timestamp = str(dt.datetime.today())
    os.makedirs(os.path.join(metrics.FIGURE_DIR, timestamp))
    
    train_loader, test_loader = load_dataset()

    train_metrics = TrainingMetrics()

    for trial in range(trials):

        if siamese:
            model = SiameseConvNet()  # if model == 'ConvNet' else SiameseMLP()
        else:
            model = ConvNet() if model == 'ConvNet' else MLP()
            
        # model.train(True) # TEST @lacoupe
        train(model, train_loader,
              learning_rate=lr, weight_decay=decay, epochs=epochs,
              metrics=train_metrics, run=trial,
              verbose=verbose)
        # model.train(False) # TEST @lacoupe
        test_metrics = TestingMetrics(model, test_loader)
        print(f"{trial:02} TEST METRICS \t {test_metrics}")
        test_metrics.plot(timestamp)

    train_metrics.plot(timestamp)


if __name__ == '__main__':
    run()
