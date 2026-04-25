"""
Author: Raphael Senn <raphaelsenn@gmx.de>

NOTE: This script is optional.
"""
from argparse import ArgumentParser, Namespace
import os

import pandas as pd
from PIL import Image

import torch

from ultralytics import YOLO


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train a GAN on MNIST")

    parser.add_argument("--root_dir", type=str, default="../../datasets/celeba/data")
    parser.add_argument("--target_dir", type=str, default="../../datasets/celeba/croped-images")

    return parser.parse_args()


def detect_and_crop_faces(
        root: str, 
        src_dir: str, 
        model_path: str,
        conf: float=0.8,
) -> None:
    """
    Very simple function to detect and crop faces using a YOLO model (face detector).
    It creates a new folder and stores the cropped faces inside that folder,
    this is done for all faces of CelebFaces (where the YOLO model detects a face w.r.t conf).

    Parameters:
    -----------
    root : str
        Root path to the CelebA dataset.

    src_dir : str
        Folder name containing the orignal images (inside CelebA).

    dst_dir : str
        Folder name storing the cropped images (inside CelebA).

    model_path : str
        Path to the weights (.pt) of a YOLO faces detector model.

    conf : float (default=0.8)
        Confidence threshold of the YOLO model.

    Personal comment:
    ----------------
    Q1: Why doing this? 
    Q2: What is the purpose of this function?

    A: In the original GAN paper (Goodfellow et al., 2014), the authors used the TorontoFacesDataset (TFD),
    which is similar to CelebFaces. HOWEVER, the TFD images are much more tightly cropped,
    making it easier for neural networks to learn the facial structures.
    This function aims to replicate the TFD dataset.
    """
    src = os.path.join(root, src_dir)
    assert  os.path.exists(src), f"{src} does not exist"

    dst = os.path.join(src, "prepro_data")
    os.makedirs(dst, exist_ok=True)
    images = sorted(os.listdir(src))

    model = YOLO(model_path)

    image_ids = []
    for img_name in images:
        image = Image.open(os.path.join(src, img_name))
        result = model.predict(image, conf=conf, verbose=False)[0]
        if not result.boxes:
            continue

        best_idx = torch.argmax(result.boxes.conf, dim=0)
        x1, y1, x2, y2 = map(int, result.boxes.xyxy[best_idx])
        image = image.crop((x1, y1, x2, y2))
        
        image.save(os.path.join(dst, img_name))
        image_ids.append(img_name)
    df_landmarks = pd.DataFrame({'image_id': image_ids})
    df_landmarks.to_csv(os.path.join(root, 'image_names.csv'), index=False)


if __name__ == "__main__":
    args = parse_args() 
    detect_and_crop_faces(args.root_dir, args.target_dir)