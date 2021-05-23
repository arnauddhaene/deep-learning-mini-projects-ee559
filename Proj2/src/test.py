from utils import load_dataset
from activation import ReLU
from linear import Linear
from train import train
from sequential import Sequential
from metrics import evaluate_accuracy


# TODO: document and extend this function into trials, etc.
def run():
    
    # Initialize model
    model = Sequential([
        Linear(2, 25), ReLU(),
        Linear(25, 25), ReLU(),
        Linear(25, 25), ReLU(),
        Linear(25, 1)])
    
    # Generate dataset
    train_input, train_target, \
        test_input, test_target = load_dataset(1000, standardize=True)
    
    # Perform weight initialization
    model.init_weights()
    
    _ = train(model, train_input, train_target)
    
    model.eval()
    
    print(f"Train accuracy: {evaluate_accuracy(model, train_input, train_target)}")
    print(f"Test accuracy: {evaluate_accuracy(model, test_input, test_target)}")
    
    
if __name__ == '__main__':
    run()
