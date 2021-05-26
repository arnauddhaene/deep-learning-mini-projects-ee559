import json

from torch import manual_seed

from torch import nn

from traintorch import train
from metrics import evaluate_accuracy
from utils import load_dataset


def train_once(model: nn.Module, trial: int = 0, verbose: int = 0) -> None:
    
    manual_seed(trial)
    
    # Generate dataset
    train_input, train_target, \
        test_input, test_target = load_dataset(1000, standardize=True)

    model.train()

    metrics = train(model, train_input, train_target, verbose=verbose)
    
    with open(f'results/metrics-trial-{trial}.json', 'w') as outfile:
        json.dump(metrics, outfile)
    
    model.train(False)

    print(f"Train accuracy: {evaluate_accuracy(model, train_input, train_target)}")
    print(f"Test accuracy: {evaluate_accuracy(model, test_input, test_target)}")
    

def run() -> None:
    
    # Initialize model
    model = nn.Sequential(
        nn.Linear(2, 25), nn.ReLU(),
        nn.Linear(25, 25), nn.Dropout(p=0.3), nn.ReLU(),
        nn.Linear(25, 25), nn.ReLU(),
        nn.Linear(25, 1))
    
    for trial in range(1):
        train_once(model, trial, 1)
    
    
if __name__ == '__main__':
    run()
