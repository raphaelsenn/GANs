from dataclasses import dataclass

import torch


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr_g: float
    lr_d: float
    betas: tuple[float, float]
    nz: int
    root_dir: str
    checkpoint_dir: str
    gan_name: str

    nc: int | None = None
    ngf: int | None = None
    ndf: int | None = None
    n_classes : int | None = None
    device: torch.device = torch.device("cpu")
    num_workers: int | None = None
    verbose: bool = True
    preview_every: int = 1
    save_every: int = 1