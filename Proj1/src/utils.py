from typing import Tuple

from dlc_practical_prologue import generate_pair_sets

import torch
from torch.utils.data import TensorDataset, DataLoader


def standardized(t: torch.tensor, mean: torch.tensor, std: torch.tensor) -> torch.tensor:
    """
    Standardize tensor following given mean and standard deviation

    Args:
        t (torch.tensor): tensor to be standardized
        mean (torch.tensor): mean
        std (torch.tensor): standard deviation

    Returns:
        torch.tensor: standardized tensor
    """
    
    return t.sub_(mean).div_(std)
  
  
def load_dataset(
    N: int = 1000, batch_size: int = 50
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset in the following format:
    * input of size [N, 2, 14, 14]: two 14x14 images (a, b)
    * target of size [1]: 1 if digit in image a <= digit in image b
    * class of size [2]: digits in images (a, b)

    Args:
        N (int, optional): Number of samples to fetch for each set.
            Defaults to 1000.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 50.

    Returns:
        Tuple[DataLoader, DataLoader]: train and test DataLoaders
    """

    train_input, train_target, train_classes, \
        test_input, test_target, test_classes = generate_pair_sets(N)
    
    mu, sigma = train_input.mean(), train_input.std()
    
    # Standardize train and test with train statistics
    train_input = standardized(train_input, mu, sigma)
    test_input = standardized(test_input, mu, sigma)

    train = TensorDataset(train_input, train_target, train_classes)
    test = TensorDataset(test_input, test_target, test_classes)
    
    return DataLoader(train, batch_size=batch_size, shuffle=True), \
        DataLoader(test, batch_size=batch_size)
