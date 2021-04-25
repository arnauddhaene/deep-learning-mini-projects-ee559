import click

from train import train
from metrics import TrainingMetrics, TestingMetrics
from models.convnet_ws import ConvNet_ws
from utils import load_dataset


@click.command()
@click.option('--epochs', default=25, help="Number of training epochs.")
@click.option('--lr', default=1e-2, help="Learning rate.")
@click.option('--decay', default=1e-3, help="Optimizer weight decay.")
@click.option('--trials', default=1, help="Number of trials to run.")
@click.option('--verbose/--no-verbose', default=False, type=bool,
              help="Print out info for debugging purposes.")
def run(epochs, lr, decay, trials, verbose):

    train_loader, test_loader = load_dataset()

    metrics = TrainingMetrics()

    for trial in range(trials):

        model = ConvNet_ws()

        train(model, train_loader,
              learning_rate=lr, weight_decay=decay, epochs=epochs,
              metrics=metrics, run=trial,
              verbose=verbose)

        print(f"{trial:02} TEST METRICS \t {TestingMetrics(model, test_loader)}")

    metrics.plot()


if __name__ == '__main__':
    run()
