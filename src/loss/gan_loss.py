"""
Author: Raphael Senn <raphaelsenn@gmx.de>

Reference:
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, Radford et al., 2016;
https://arxiv.org/abs/1511.06434

Generative Adversarial Networks, Goodfellow et al. 2014
https://arxiv.org/abs/1406.2661
"""
import torch
import torch.nn as nn


class GeneratorLoss(nn.Module):
    """
    Implementation of the (non-saturating) generator loss as described in the original GAN paper.

    Objective:
    max_G E[log D(G(noise))]
    <=>
    min_G -E[log D(G(noise))]
    <=> 
    min_G BCE(D(G(noise)), 1)

    Reference:
    Generative Adversarial Networks, Goodfellow et al. 2014
    https://arxiv.org/abs/1406.2661
    """  
    def __init__(self) -> None:
        super().__init__() 
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, fake_score: torch.Tensor) -> torch.Tensor:
        return self.bce_with_logits(fake_score, torch.ones_like(fake_score))


class DiscriminatorLoss(nn.Module):
    """
    Implementation of the descriminator loss as described in the original gan paper.

    Objective:
    max_D E[log(D(real_img))] + E_z[log(1 - D(G(noise)))]
    <=>
    min_D -(E[log(D(real_img))] + E[log(1 - D(G(noise)))])
    <=>
    min_D BCE(D(real_img), 1) + BCE(D(G(noise)), 0)

    Reference:
    Generative Adversarial Networks, Goodfellow et al. 2014
    https://arxiv.org/abs/1406.2661
    """ 
    def __init__(self) -> None:
        super().__init__()  
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
        real = torch.ones_like(real_score)
        fake = torch.zeros_like(fake_score)
        return self.bce_with_logits(real_score, real) + self.bce_with_logits(fake_score, fake)