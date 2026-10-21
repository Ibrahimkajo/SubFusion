"""Run the released SubFusion MSRS checkpoint on paired visible/IR images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from subfusion.io import paired_paths, read_paired_image, to_model_input, to_uint8
from subfusion.model import load_msrs_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Folder of horizontally paired images: visible left, IR right.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "weights" / "subfusion_msrs_full_rank1_70e.pth"))
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_msrs_checkpoint(args.checkpoint, device)
    outputs = Path(args.output_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    paths = paired_paths(args.input_dir)[: args.limit]
    with torch.inference_mode():
        for index, path in enumerate(paths, start=1):
            visible, infrared = read_paired_image(path, args.image_size)
            fused = model(to_model_input(visible, device), to_model_input(infrared, device))
            Image.fromarray(to_uint8(fused)).save(outputs / f"{path.stem}.png")
            print(f"{index}/{len(paths)} {path.name}")


if __name__ == "__main__":
    main()
