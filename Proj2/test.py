import json

from torch import manual_seed

from flame import nn

from train import train
from metrics import evaluate_accuracy
from utils import load_dataset


def train_once(model: nn.Module, trial: int = 0, **kwargs) -> None:
    
    # Initialize model
    manual_seed(trial)
    
    # Generate dataset
    train_input, train_target, \
        test_input, test_target = load_dataset(1000, standardize=True)
    
    # Perform weight initialization
    model.init_weights()

    model.train()

    train(model, train_input, train_target, **kwargs)
       
    final_train = evaluate_accuracy(model, train_input, train_target).item()
    final_test = evaluate_accuracy(model, test_input, test_target).item()

    print(f"Train accuracy: {final_train}")
    print(f"Test accuracy: {final_test}")


def run() -> None:
    
    trial_metrics = []
    
    # TODO: add CrossEntropy
    model_configs = [
        dict(optim='SGD', crit='MSE', learning_rate=1e-1),
        dict(optim='Adam', crit='MSE', learning_rate=1e-3),
        dict(optim='Adagrad', crit='MSE', learning_rate=1e-3),
    ]
    
    for config in model_configs:
    
        # Initialize model
        model = nn.Sequential([
            nn.Linear(2, 25), nn.ReLU(),
            nn.Linear(25, 25), nn.Dropout(p=0.3), nn.ReLU(),
            nn.Linear(25, 25), nn.ReLU(),
            nn.Linear(25, 1)])
        
        for trial in range(15):
            train_once(model, trial, metrics=trial_metrics, **config, verbose=0)

    with open('results/metrics.json', 'w') as outfile:
        json.dump(trial_metrics, outfile)
    
    
if __name__ == '__main__':
    run()
