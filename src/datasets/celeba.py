import os
from typing import Callable, Any

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class CelebA(Dataset):
    """
    PyTorch dataset wrapper for CelebFaces (CelebA).

    Args:    
    ---------
    root : str
        Root img directory of CelebA dataset.

    Reference:
    ---------
    CelebFaces Attributes Dataset (CelebA):
    Liu, Ziwei, et al. "Deep learning face attributes in the wild."
    Proceedings of the IEEE International Conference on Computer Vision. 2015.
    https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    """
    def __init__(
            self, 
            root_dir: str,
            img_file: str,
            transform: Callable|None=None
    ) -> None:
        assert os.path.isdir(root_dir), f"Directory {root_dir} does not exist."
        assert os.path.isfile(img_file), f"File {img_file} does not exist."
        
        self.root_dir = root_dir
        self.df = pd.read_csv(img_file)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int | torch.Tensor) -> Image.Image | torch.Tensor:
        if isinstance(index, torch.Tensor): 
            index = index.item()
        
        img_name = self.df.iloc[index, 0]
        image_path = os.path.join(self.root_dir, img_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, index