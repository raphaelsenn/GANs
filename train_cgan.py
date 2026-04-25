from argparse import ArgumentParser, Namespace
import random

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.config import TrainConfig
from src.trainer.cgan_trainer import CGANTrainer
from src.models.cgan import CGANGenerator, CGANDiscriminator


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train a CGAN on MNIST")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr_g", type=float, default=0.002)
    parser.add_argument("--lr_d", type=float, default=0.002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)

    # Model
    # Original gan hyperparameters from: https://arxiv.org/abs/1411.1784
    parser.add_argument("--nz", type=int, default=100, help="Noise dimension")
    parser.add_argument("--image_dim", type=int, default=784, help="Image dimension")
    parser.add_argument("--nc", type=int, default=10, help="Num classes")

    # Dataset and gan type (i.e. vanilla-gan, cgan, dcgan)
    parser.add_argument("--root_dir", type=str, default="./MNIST")
    parser.add_argument("--checkpoint_dir", type=str, default="./cgan-mnist/")
    parser.add_argument("--gan_name", type=str, default="cgan-mnist")

    # Misc
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--preview_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)

    return parser.parse_args()


def build_config(args: Namespace) -> TrainConfig:
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        betas=(args.beta1, args.beta2),
        nz=args.nz,
        nc=args.nc,
        device=torch.device(args.device),
        verbose=args.verbose,
        root_dir=args.root_dir,
        checkpoint_dir=args.checkpoint_dir,
        gan_name=args.gan_name,
        preview_every=args.preview_every,
        save_every=args.save_every,
    )


def build_dataloader(cfg: TrainConfig) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    dataset = MNIST(
        root=cfg.root_dir,
        train=True,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )


def set_seeds(seed: int) -> None:
    """Sets random seeds for reproducability for both CPU.""" 
    random.seed(seed) 
    torch.manual_seed(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    cfg = build_config(args)

    dataloader = build_dataloader(cfg)

    generator = CGANGenerator(cfg.nz)
    discriminator = CGANDiscriminator(args.image_dim)

    trainer = CGANTrainer(cfg, generator, discriminator)
    trainer.train(dataloader)


if __name__ == "__main__":
    main()