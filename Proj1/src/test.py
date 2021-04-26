import click

from train import train
from metrics import TrainingMetrics, TestingMetrics
from models.convnet import ConvNet
from models.siamese_convnet import SiameseConvNet
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
@click.option('--verbose/--no-verbose', default=False, type=bool,
              help="Print out info for debugging purposes.")
def run(model, siamese, epochs, lr, decay, trials, verbose):

    train_loader, test_loader = load_dataset()

    train_metrics = TrainingMetrics()

    for trial in range(trials):

        if siamese:
            model = SiameseConvNet()  # if model == 'ConvNet' else MLP()
        else:
            model = ConvNet()  # if model == 'ConvNet' else SiameseMLP()
            
        train(model, train_loader,
              learning_rate=lr, weight_decay=decay, epochs=epochs,
              metrics=train_metrics, run=trial,
              verbose=verbose)
        
        test_metrics = TestingMetrics(model, test_loader)
        print(f"{trial:02} TEST METRICS \t {test_metrics}")
        test_metrics.plot()

    train_metrics.plot()


if __name__ == '__main__':
    run()
