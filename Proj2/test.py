from torch import manual_seed

from flame import nn
from train import train
from metrics import evaluate_accuracy
from utils import load_dataset


# TODO: document and extend this function into trials, etc.
def run():
    
    manual_seed(3)
    
    # Initialize model
    model = nn.Sequential([
        nn.Linear(2, 25), nn.ReLU(),
        nn.Linear(25, 100), nn.Dropout(p=0.3), nn.ReLU(),
        nn.Linear(100, 25), nn.ReLU(),
        nn.Linear(25, 1)])
    
    # Generate dataset
    train_input, train_target, \
        test_input, test_target = load_dataset(1000, standardize=True)
    
    # Perform weight initialization
    model.init_weights()

    model.train()

    _ = train(model, train_input, train_target)
    
    model.test()

    print(f"Train accuracy: {evaluate_accuracy(model, train_input, train_target)}")
    print(f"Test accuracy: {evaluate_accuracy(model, test_input, test_target)}")
    
    
if __name__ == '__main__':
    run()
