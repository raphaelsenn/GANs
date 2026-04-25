import torch
import torch.nn as nn

from src.models.base import Generator, Discriminator
from src.utils.maxout_layer import Maxout


class CGANGenerator(Generator):
    """
    Implementation of the conditional generator (MNIST Mirza et al., 2014).

    Reference:
    Conditional Generative Adversarial Nets, Mirza et al. 2014;
    https://arxiv.org/abs/1411.1784
    """
    def __init__(
            self,
            nz: int=100,
            nc: int=10,
            z_hidden_dim: int=200,
            c_hidden_dim: int=1000,
            hidden_dim: int=1200,
            output_dim: int=784
    ) -> None:
        super().__init__(nz)

        # Read more about this neural net architecture here:
        # https://arxiv.org/abs/1411.1784

        # [B, nz] -> [B, z_hidden_dim]
        self.branch_noise = nn.Sequential(
            nn.Linear(nz, z_hidden_dim),
            nn.ReLU(True)
        )

        # [B, nc] -> [B, c_hidden_dim]
        self.branch_classes = nn.Sequential(
            nn.Linear(nc, c_hidden_dim),
            nn.ReLU(True),
        ) 
        
        self.out = nn.Sequential(
            # [B, z_hidden_dim + c_hidden_dim] -> [B, hidden_dim] 
            nn.Linear(z_hidden_dim + c_hidden_dim, hidden_dim),
            nn.ReLU(True),
            
            # [B, hidden_dim] -> [B, output_dim] 
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def forward(self, noise: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        hz = self.branch_noise(noise)
        hc = self.branch_classes(classes)
        h = torch.cat([hz, hc], dim=-1)
        return self.out(h)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(
                    m.weight, -0.005, 0.005
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class CGANDiscriminator(Discriminator):
    """
    Implementation of the conditional discriminator (MNIST, Mirza et al., 2014).

    Reference:
    Conditional Generative Adversarial Nets, Mirza et al., 2014;
    https://arxiv.org/abs/1411.1784
    """  
    def __init__(
            self,
            image_dim: int=784,
            image_hidden_dim: int=240,
            image_pieces: int=5,
            image_in_dropout: float=0.2,
            nc: int=10,
            c_hidden_dim: int=50,
            c_pieces: int=5,
            c_in_dropout: float=0.2,
            hidden_dim: int=240,
            hidden_pieces: int=4,
            hidden_dropout: float=0.5
    ) -> None:
        super().__init__()

        # Read more about this neural net architecture here:
        # https://arxiv.org/abs/1411.1784

        # [B, image_dim] -> [B, image_hidden_dim]
        self.branch_image = nn.Sequential(
            nn.Dropout(image_in_dropout),
            Maxout(image_dim, image_hidden_dim, image_pieces)
        )
        
        # [B, nc] -> [B, c_hidden_dim]
        self.branch_classes = nn.Sequential(
            nn.Dropout(c_in_dropout),
            Maxout(nc, c_hidden_dim, c_pieces)
        )

        self.out = nn.Sequential(
            # [B, image_hidden_dim + c_hidden_dim] -> [B, hidden_dim]
            nn.Dropout(hidden_dropout),
            Maxout(image_hidden_dim + c_hidden_dim, hidden_dim, hidden_pieces),
            nn.Dropout(hidden_dropout),

            # [B, hidden_dim] -> [B, 1]
            nn.Linear(hidden_dim, 1),
        )
        self._initialize_weights()

    def forward(self, image: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        hi = self.branch_image(image)
        hc = self.branch_classes(classes)
        h = torch.cat([hi, hc], dim=-1)
        return self.out(h)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.005, 0.005)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)