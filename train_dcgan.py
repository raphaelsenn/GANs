from argparse import ArgumentParser, Namespace
import random

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.config import TrainConfig
from src.trainer.gan_trainer import GANTrainer
from src.models.dcgan_v1 import DCGANGenerator, DCGANDiscriminator
from src.datasets.celeba import CelebA


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train a DCGAN on CelebA")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.99)

    # Model
    parser.add_argument("--nz", type=int, default=100, help="Noise dimension")
    parser.add_argument("--ngf", type=int, default=128)
    parser.add_argument("--ndf", type=int, default=128)
    parser.add_argument("--channels_img", type=int, default=3)

    # Dataset and gan type (i.e. vanilla-gan, cgan, dcgan)
    parser.add_argument("--root_dir", type=str, default="../../datasets/celeba/data/")
    parser.add_argument("--path_landmarks_csv", type=str, default="../../datasets/celeba/landmarks.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="./dcgan-celeba/")
    parser.add_argument("--gan_name", type=str, default="dcgan-celeba")

    # Misc
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
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
        ngf=args.ngf,
        ndf=args.ndf,
        device=torch.device(args.device),
        num_workers=args.num_workers,
        verbose=args.verbose,
        root_dir=args.root_dir,
        checkpoint_dir=args.checkpoint_dir,
        gan_name=args.gan_name,
        preview_every=args.preview_every,
        save_every=args.save_every,
    )


def build_dataloader(cfg: TrainConfig, args: Namespace) -> DataLoader:
    W, H = 64, 64
    transform = transforms.Compose([
        transforms.Resize(size=(W, H)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CelebA(args.root_dir, args.path_landmarks_csv, transform)

    dataloader = DataLoader(
        dataset, 
        cfg.batch_size, 
        shuffle=True,
        num_workers=cfg.num_workers, 
    )
    return dataloader

def set_seeds(seed: int) -> None:
    """Sets random seeds for reproducability for both CPU.""" 
    random.seed(seed) 
    torch.manual_seed(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    cfg = build_config(args)

    dataloader = build_dataloader(cfg, args)

    generator = DCGANGenerator(cfg.nz, cfg.ngf)
    discriminator = DCGANDiscriminator(cfg.ndf, args.channels_img)

    trainer = GANTrainer(cfg, generator, discriminator)
    trainer.train(dataloader)


if __name__ == "__main__":
    main()