from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.config import TrainConfig


class Preview:
    def __init__(
        self,
        generator: nn.Module,
        cfg: TrainConfig,
        n_rows: int = 6,
        n_cols: int = 6,
        dpi: int = 100,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        self.generator = generator
        self.cfg = cfg

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_images = n_rows * n_cols

        self.dpi = dpi
        self.image_size = image_size

        self.n_steps = 0

        self.save_dir = Path(f"{cfg.gan_name}_preview")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.noise = self.sample_noise(self.n_images)

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.rand(
            size=(batch_size, self.cfg.nz),
            device=self.cfg.device,
        ) * 2.0 - 1.0

    @torch.no_grad()
    def generate(self) -> torch.Tensor:
        return self.generator(self.noise)

    def _to_image(self, img: torch.Tensor):
        """
        Converts one generated image tensor to a NumPy image usable by imshow.

        Supports:
        - [C, H, W]
        - [H, W]
        - [D]
        """
        img = img.detach().cpu()

        # If generator outputs tanh range [-1, 1] map to [0, 1]
        img = (img + 1.0) / 2.0
        img = img.clamp(0.0, 1.0)

        # Flattened image, i.e. MNIST [784]
        if img.ndim == 1:
            if self.image_size is None:
                side = int(img.numel() ** 0.5)
                img = img.reshape(side, side)
            else:
                h, w = self.image_size
                img = img.reshape(h, w)

            return img.numpy(), "gray"

        # Already grayscale [H, W]
        if img.ndim == 2:
            return img.numpy(), "gray"

        # Channel-first image [C, H, W]
        if img.ndim == 3:
            c, h, w = img.shape

            if c == 1:
                img = img.squeeze(0)
                return img.numpy(), "gray"

            if c == 3:
                img = img.permute(1, 2, 0)
                return img.numpy(), None

            raise ValueError(f"Unsupported channel count: {c}")

        raise ValueError(f"Unsupported image shape: {tuple(img.shape)}")

    def preview(self) -> None:
        was_training = self.generator.training
        self.generator.eval()

        fake_imgs = self.generate()

        if was_training:
            self.generator.train()

        fig_width = self.n_cols
        fig_height = self.n_rows

        fig, axes = plt.subplots(
            nrows=self.n_rows,
            ncols=self.n_cols,
            figsize=(fig_width, fig_height),
            dpi=self.dpi,
        )

        fig.subplots_adjust(
            hspace=0,
            wspace=0,
            top=1,
            bottom=0,
            left=0,
            right=1,
        )

        axes = axes.flatten()

        for i in range(self.n_images):
            img, cmap = self._to_image(fake_imgs[i])

            axes[i].imshow(img, cmap=cmap)
            axes[i].axis("off")

        save_path = self.save_dir / f"{self.n_steps:06d}.png"
        fig.savefig(save_path, dpi=self.dpi, pad_inches=0)
        plt.close(fig)

        self.n_steps += 1