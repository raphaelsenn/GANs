"""
Author: Raphael Senn <raphaelsenn@gmx.de>
Initial coding: 2025-07-14
"""
import torch
import torch.nn as nn

from src.models.base import Generator, Discriminator
from src.utils.maxout_layer import Maxout


class GANGenerator(Generator):
    """
    Implementation of the MNIST-Generator from the original GAN paper.

    Reference:
    Generative Adversarial Networks, Goodfellow et al. 2014
    https://arxiv.org/abs/1406.2661
    """ 
    def __init__(
            self, 
            nz: int=100, 
            hidden_dim: int=1200, 
            output_dim: int=784,
            uniform_init: tuple[float, float] = (-0.05, 0.05)
        ) -> None:
        super().__init__(nz)
        
        self.uniform_init = uniform_init

        self.net = nn.Sequential(
            # [B, nz] -> [B, hidden_dim] 
            nn.Linear(nz, hidden_dim),
            nn.ReLU(True),
            
            # [B, hidden_dim] -> [B, hidden_dim] 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            
            # [B, hidden_dim] -> [B, output_dim] 
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.net(noise)
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(
                    m.weight, self.uniform_init[0], self.uniform_init[1]
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class GANDiscriminator(Discriminator):
    """
    Implementation of the MNIST-Discriminator from the original GAN paper.

    NOTE: Output are raw logits.

    Reference:
    Generative Adversarial Networks, Goodfellow et al. 2014
    https://arxiv.org/abs/1406.2661
    """  
    def __init__(
            self, 
            input_dim: int=784,
            hidden_dim: int=240,
            num_pieces: int=5,
            input_p: float=0.2,
            hidden_p: float=0.5,
            uniform_init: tuple[float, float] = (-0.005, 0.005)
        ) -> None:
        super().__init__()
        self.uniform_init = uniform_init

        self.net = nn.Sequential(
            # [B, input_dim] -> [B, hidden_dim]
            nn.Dropout(input_p), 
            Maxout(input_dim, hidden_dim, num_pieces),
            nn.Dropout(hidden_p),
            
            # [B, hidden_dim] -> [B, hidden_dim]
            Maxout(hidden_dim, hidden_dim, num_pieces),
            nn.Dropout(hidden_p),
            
            # [B, hidden_dim] -> [B, 1]
            nn.Linear(hidden_dim, 1), 
        )
        self._initialize_weights()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Output are raw logits.""" 
        return self.net(image)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(
                    m.weight, self.uniform_init[0], self.uniform_init[1]
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)