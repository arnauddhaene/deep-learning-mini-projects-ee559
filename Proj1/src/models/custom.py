import torch
from torch import nn


class SizeableModule(nn.Module):
    
    def __init(self):
        super().__init__()
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        raise NotImplementedError
    
    def param_count(self) -> int:
        """Parameter counter"""
        return sum(param.numel() for param in self.parameters())


class NamedModule(nn.Module):
    
    def __init(self):
        super().__init__()
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        raise NotImplementedError
    
    def __str__(self):
        raise NotImplementedError
