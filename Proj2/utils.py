import math
from typing import Tuple

from torch import empty, Tensor


def generate(samples: int) -> Tuple[Tensor, Tensor]:
    """
    Generate points sampled uniformly in [0, 1]^2 space
    Each point is associated to a label 0 if outside the wanted disk and 1 if inside
    The wanted disk is centered at (0.5, 0.5) and has radius 1 / √(2π)

    Args:
        samples (int): Number of sample points to generate

    Returns:
        Tuple[Tensor, Tensor]: points and their attributed labels
    """
    
    data = empty(samples, 2).uniform_(0, 1)
    
    target = data.sub(0.5).pow(2).sum(1).sub(1 / (2 * math.pi)).sign().mul(-1).add(1).div(2).long()
    
    return data, target


def load_dataset(
    samples: int, standardize: bool = True
) -> Tuple[Tensor, Tensor, Tensor, Tensor, float, float]:
    """
    Load the wanted dataset as explicited in the mini-project description

    Args:
        samples (int): Number of samples to generate in train and test sets.
        standardize (bool, optional): standardize data w.r.t. to training set. Defaults to True.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, float, float]: {train, test} {input, target}, mu, sig
    """
    
    train_input, train_target = generate(samples)
    test_input, test_target = generate(samples)
    
    if standardize:
        mu, sigma = train_input.mean(), train_input.std()
        
        # Standardize train and test with train statistics
        train_input = standardized(train_input, mu, sigma)
        test_input = standardized(test_input, mu, sigma)
        return train_input, train_target, test_input, test_target, mu, sigma
    else: 
        return train_input, train_target, test_input, test_target

        
        
def standardized(t: Tensor, mean: Tensor, std: Tensor) -> Tensor:
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
