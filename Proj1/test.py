from models.convnet import ConvNet
import click

from models.convnet import ConvNet
from utils import load_dataset
from train import train
from metrics import evaluate_accuracy

@click.command()

@click.option('--epochs', default=25, help="Number of training epochs")
@click.option('--lr', default=1e-2, help="Learning rate")
@click.option('--decay', default=1e-3, help="Optimizer weight decay")
@click.option('--verbose', default=False, help="Print out info for debugging")

def run(epochs, lr, decay, verbose):
    
    print('model init')
    
    model = ConvNet()
    
    train_loader, test_loader = load_dataset()
    
    print('data fetched')
    
    metrics = train(model, train_loader, 
                    learning_rate=lr, weight_decay=decay, epochs=epochs,
                    verbose=verbose)
    
    print('trained')
    
    metrics.plot()
    
    print(f"Test accuracy {evaluate_accuracy(model, test_loader) * 100:06.3f}")


if __name__ == '__main__':
    run()