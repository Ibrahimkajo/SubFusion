"""Dataset support for horizontally paired visible/infrared images."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .io import IMAGE_SUFFIXES


class PairedFusionDataset(Dataset):
    """Loads paired files with visible RGB on the left and IR on the right."""

    def __init__(self, root: str | Path, load_size: int = 480, crop_size: int = 480, flip: bool = True):
        self.paths = sorted(path for path in Path(root).rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if not self.paths:
            raise FileNotFoundError(f"No paired images found in {root}")
        self.load_size, self.crop_size, self.flip = load_size, crop_size, flip

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.paths[index]).convert("RGB")
        split = image.width // 2
        visible = image.crop((0, 0, split, image.height)).resize((self.load_size, self.load_size), Image.Resampling.BICUBIC)
        infrared = image.crop((split, 0, image.width, image.height)).resize((self.load_size, self.load_size), Image.Resampling.BICUBIC)
        if self.load_size > self.crop_size:
            left = random.randint(0, self.load_size - self.crop_size)
            top = random.randint(0, self.load_size - self.crop_size)
            crop = (left, top, left + self.crop_size, top + self.crop_size)
            visible, infrared = visible.crop(crop), infrared.crop(crop)
        if self.flip and random.random() < 0.5:
            visible, infrared = visible.transpose(Image.Transpose.FLIP_LEFT_RIGHT), infrared.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return self._tensor(visible), self._tensor(infrared)

    @staticmethod
    def _tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)
        return torch.from_numpy(array.copy()).permute(2, 0, 1).div(127.5).sub(1.0)
