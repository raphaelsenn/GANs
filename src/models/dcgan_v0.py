"""
Author: Raphael Senn <raphaelsenn@gmx.de>

Reference:
Unsupervised Representation Learning with
Deep Convolutional Generative Adversarial Networks, Radford et al., 2016;
https://arxiv.org/abs/1511.06434

Generative Adversarial Networks, Goodfellow et al. 2014
https://arxiv.org/abs/1406.2661
"""
import torch
import torch.nn as nn
from src.models.base import Generator, Discriminator


class DCGANGenerator(Generator):
    """
    Implementation of the DCGAN-Generator.

    Reference:
    Unsupervised Representation Learning with 
    Deep Convolutional Generative Adversarial Networks, Radford et al., 2016;
    https://arxiv.org/abs/1511.06434
    """
    def __init__(
            self, 
            nz: int=100,
            ngf: int=128,
            channels_img: int=3,
    ) -> None:
        super().__init__(nz) 
        self.ngf = ngf
        self.channels_img = channels_img

        self.net = nn.Sequential(
            # [N, nz, 1, 1] -> [N, 8*ngf, 4, 4] 
            nn.ConvTranspose2d(nz, 8*ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(8 * ngf),
            nn.ReLU(True),

            # [N, 8*ngf, 4, 4] -> [N, 4*ngf, 8, 8]
            nn.ConvTranspose2d(8*ngf, 4*ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*ngf),
            nn.ReLU(True),

            # [N, 4*ngf, 8, 8] -> [N, 2*ngf, 16, 16]
            nn.ConvTranspose2d(4*ngf, 2*ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*ngf),
            nn.ReLU(True),

            # [N, 2*ngf, 16, 16] -> [N, ngf, 32, 32]
            nn.ConvTranspose2d(2*ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # [N, ngf, 32, 32] -> [N, channels_img, 64, 64]
            nn.ConvTranspose2d(ngf, channels_img, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self._initialize_weights()

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        # [N, nz] -> [N, nz, 1, 1]
        if noise.ndim == 2:
            noise = noise.view((-1, self.nz, 1, 1))
        return self.net(noise)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class DCGANDiscriminator(Discriminator):
    """
    Implementation of the DCGAN-Discriminator.
 
    Reference:
    Unsupervised Representation Learning with
    Deep Convolutional Generative Adversarial Networks, Radford et al., 2016;
    https://arxiv.org/abs/1511.06434
    """ 
    def __init__(
            self,
            ndf: int=128,
            channels_img: int=3,
        ) -> None:
        super().__init__()
        self.ndf = ndf
        self.channels_img = channels_img

        self.net = nn.Sequential(
            # [N, channels_img, 64, 64] -> [N, ndf, 32, 32]
            nn.Conv2d(channels_img, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            
            # [N, ndf, 32, 32] -> [N, 2*ndf, 16, 16]
            nn.Conv2d(ndf, 2*ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*ndf),
            nn.LeakyReLU(0.2, True),
            
            # [N, 2*ndf, 16, 16] -> [N, 4*ndf, 8, 8]
            nn.Conv2d(2*ndf, 4*ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4*ndf),
            nn.LeakyReLU(0.2, True),

            # [N, 4*ndf, 8, 8] -> [N, 8*ndf, 4, 4]
            nn.Conv2d(4*ndf, 8*ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8*ndf),
            nn.LeakyReLU(0.2, True),
            
            # [N, 8*ndf, 4, 4] ->  [N, 1, 1, 1]
            nn.Conv2d(8*ndf, 1, 4, 1, 0, bias=False)
        )
        self._initialize_weights()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)