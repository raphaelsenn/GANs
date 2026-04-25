import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.trainer.base import GANTrainerBase


class CGANTrainer(GANTrainerBase):
    """Simple CGAN trainer."""
    def __init__(
        self,
        cfg: TrainConfig,
        generator: nn.Module,
        discriminator: nn.Module,
    ) -> None:
        super().__init__(cfg, generator, discriminator)

    def train_discriminator(
            self, 
            real_img: torch.Tensor, 
            classes: torch.Tensor
    ) -> torch.Tensor:
        self.discriminator.train() 
        
        # Read more here: https://arxiv.org/abs/1406.2661
        batch_size = real_img.size(0)

        noise = self.sample_noise(batch_size)

        with torch.no_grad():
            fake_img = self.generator(noise, classes)

        real_score = self.discriminator(real_img, classes)
        fake_score = self.discriminator(fake_img, classes)

        loss_d = self.criterion_d(real_score, fake_score)

        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()

        return loss_d.detach()

    def train_generator(self, batch_size: int, classes: torch.Tensor) -> torch.Tensor:
        self.generator.train() 
        
        # Read more here: https://arxiv.org/abs/1406.2661
        noise = self.sample_noise(batch_size)
        fake_img = self.generator(noise, classes)
        fake_score = self.discriminator(fake_img, classes)

        loss_g = self.criterion_g(fake_score)

        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.detach()

    def train_one_epoch(self, dataloader: DataLoader) -> None:
        total_loss_g = 0.0
        total_loss_d = 0.0

        for step, (real_img, classes) in enumerate(dataloader):
            real_img = real_img.to(self.cfg.device)
            classes = F.one_hot(classes, self.cfg.nc).float().to(self.cfg.device)
            batch_size = int(real_img.size(0))

            loss_d = self.train_discriminator(real_img, classes)
            loss_g = self.train_generator(batch_size, classes)

            total_loss_d += loss_d.item() * batch_size
            total_loss_g += loss_g.item() * batch_size

        n_samples = len(dataloader.dataset)

        self.report["Loss_G"].append(total_loss_g / n_samples)
        self.report["Loss_D"].append(total_loss_d / n_samples)