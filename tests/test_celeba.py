from pathlib import Path

import pytest

from src.datasets.celeba import CelebA


ROOT_DIR = Path("../../datasets/celeba/data/")
PATH_LANDMARKS_FILE = Path("../../datasets/celeba/landmarks.csv")


class TestCelebA:
    def test_all_images_exist(self) -> None:
        dataset = CelebA(root_dir=str(ROOT_DIR), img_file=str(PATH_LANDMARKS_FILE))

        for i in range(len(dataset)):
            img_name = dataset.df.iloc[i, 0].strip()
            img_path = ROOT_DIR / img_name

            assert img_path.is_file(), f"Missing image: {img_path}"

    def test_dataset_length(self) -> None:
        dataset = CelebA(root_dir=str(ROOT_DIR), img_file=str(PATH_LANDMARKS_FILE))
        assert len(dataset) == 202599