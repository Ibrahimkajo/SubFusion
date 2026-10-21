"""Paired-image I/O used for reproducible SubFusion inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def paired_paths(root: str | Path) -> list[Path]:
    root = Path(root)
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No paired images found in {root}")
    return paths


def read_paired_image(path: str | Path, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a horizontally paired file: left visible RGB, right IR RGB."""
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width < 2:
        raise ValueError(f"Paired image is too narrow: {path}")
    split = width // 2
    resample = Image.Resampling.LANCZOS
    visible = image.crop((0, 0, split, height)).resize((image_size, image_size), resample)
    infrared = image.crop((split, 0, width, height)).resize((image_size, image_size), resample)
    return np.asarray(visible), np.asarray(infrared)


def to_model_input(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(127.5).sub(1.0).unsqueeze(0).to(device)


def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu()[0].float().permute(1, 2, 0).numpy()
    return np.clip(np.rint((image + 1.0) * 127.5), 0, 255).astype(np.uint8)
