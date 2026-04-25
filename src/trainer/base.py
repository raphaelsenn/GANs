from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.loss.gan_loss import GeneratorLoss, DiscriminatorLoss
from src.utils.preview import Preview


class GANTrainerBase(ABC):
    """Simple GAN trainer."""
    def __init__(
        self,
        cfg: TrainConfig,
        generator: nn.Module,
        discriminator: nn.Module,
    ) -> None:
        self.cfg = cfg

        self.generator = generator.to(cfg.device)
        self.discriminator = discriminator.to(cfg.device)

        self.criterion_g = GeneratorLoss()
        self.criterion_d = DiscriminatorLoss()

        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=cfg.lr_g,
            betas=cfg.betas,
        )

        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=cfg.lr_d,
            betas=cfg.betas,
        )

        self.report: dict[str, list[float]] = {
            "Loss_G": [],
            "Loss_D": [],
        }

        self.preview = Preview(self.generator, self.cfg)

    @abstractmethod 
    def train_generator(*args, **kwargs) -> float:
        raise NotImplementedError

    @abstractmethod 
    def train_discriminator(*args, **kwargs) -> float:
        raise NotImplementedError
    
    @abstractmethod 
    def train_one_epoch(dataloader: DataLoader) -> None:
        raise NotImplementedError

    def train(self, dataloader: DataLoader) -> None:
        for epoch in range(self.cfg.epochs):
            self.train_one_epoch(dataloader)
            self.checkpoint()

            if self.cfg.verbose:
                print(
                    f"Epoch: {epoch + 1}/{self.cfg.epochs}\t"
                    f"Loss_G: {self.report['Loss_G'][-1]:.4f}\t"
                    f"Loss_D: {self.report['Loss_D'][-1]:.4f}"
                )

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        # Read more here: https://arxiv.org/abs/1406.2661
        return torch.rand(
            size=(batch_size, self.cfg.nz),
            device=self.cfg.device,
        ) * 2.0 - 1.0 
    
    def checkpoint(self) -> None:
        checkpoint_dir = Path(self.cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        generator_name = (
            f"generator_{self.cfg.gan_name}_"
            f"ngf{self.cfg.ngf}_"
            f"epochs{self.cfg.epochs}.pth"
        )

        discriminator_name = (
            f"discriminator_{self.cfg.gan_name}_"
            f"ndf{self.cfg.ndf}_"
            f"epochs{self.cfg.epochs}.pth"
        )

        torch.save(
            self.generator.state_dict(),
            checkpoint_dir / generator_name,
        )

        torch.save(
            self.discriminator.state_dict(),
            checkpoint_dir / discriminator_name,
        )

        report_name = f"{self.cfg.gan_name}_report.csv"

        pd.DataFrame.from_dict(self.report).to_csv(
            checkpoint_dir / report_name,
            index=False,
        )