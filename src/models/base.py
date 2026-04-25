from abc import ABC, abstractmethod

import torch.nn as nn


class Generator(ABC, nn.Module):
    """Generator interface.""" 
    def __init__(self, nz: int=100) -> None:
        super().__init__() 
        self.nz = nz

    @abstractmethod 
    def _initialize_weights(self) -> None:
        raise NotImplementedError


class Discriminator(ABC, nn.Module):
    """Discriminator interface.""" 
    def __init__(self) -> None:
        super().__init__() 

    @abstractmethod 
    def _initialize_weights(self) -> None:    
        raise NotImplementedError