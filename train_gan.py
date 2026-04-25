from argparse import ArgumentParser, Namespace
import random

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.config import TrainConfig
from src.trainer.gan_trainer import GANTrainer
from src.models.gan import GANGenerator, GANDiscriminator


from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train a vanilla GAN on MNIST")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--lr_g", type=float, default=2e-3)
    parser.add_argument("--lr_d", type=float, default=2e-3)

    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.99)

    # Generator
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--generator_hidden_dim", type=int, default=1200)
    parser.add_argument("--output_dim", type=int, default=784)

    parser.add_argument("--generator_init_low", type=float, default=-0.05)
    parser.add_argument("--generator_init_high", type=float, default=0.05)

    # Discriminator
    parser.add_argument("--input_dim", type=int, default=784)
    parser.add_argument("--discriminator_hidden_dim", type=int, default=240)

    parser.add_argument("--num_pieces", type=int, default=5)
    parser.add_argument("--input_p", type=float, default=0.2)
    parser.add_argument("--hidden_p", type=float, default=0.5)

    parser.add_argument("--discriminator_init_low", type=float, default=-0.005)
    parser.add_argument("--discriminator_init_high", type=float, default=0.005)

    # Dataset / experiment
    parser.add_argument("--root_dir", type=str, default="./MNIST")
    parser.add_argument("--checkpoint_dir", type=str, default="./vanilla-gan-mnist")
    parser.add_argument("--gan_name", type=str, default="vanilla-gan-mnist")

    # Misc
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--preview_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_verbose", dest="verbose", action="store_false")
    parser.set_defaults(verbose=True)

    return parser.parse_args()


def build_config(args: Namespace) -> TrainConfig:
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        betas=(args.beta1, args.beta2),
        nz=args.nz,
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

    generator = GANGenerator(
        nz=args.nz,
        hidden_dim=args.generator_hidden_dim,
        output_dim=args.output_dim,
        uniform_init=(
            args.generator_init_low,
            args.generator_init_high,
        ),
    )

    discriminator = GANDiscriminator(
        input_dim=args.input_dim,
        hidden_dim=args.discriminator_hidden_dim,
        num_pieces=args.num_pieces,
        input_p=args.input_p,
        hidden_p=args.hidden_p,
        uniform_init=(
            args.discriminator_init_low,
            args.discriminator_init_high,
        ),
    )

    trainer = GANTrainer(cfg, generator, discriminator)
    trainer.train(dataloader)


if __name__ == "__main__":
    main()