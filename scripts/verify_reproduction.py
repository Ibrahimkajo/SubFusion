"""Verify that this standalone release reproduces a prior fused-image export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from subfusion.io import paired_paths, read_paired_image, to_model_input, to_uint8
from subfusion.model import load_msrs_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "weights" / "subfusion_msrs_full_rank1_70e.pth"))
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--report", default="reproduction_report.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_msrs_checkpoint(args.checkpoint, device)
    maximum, total, pixels, checked = 0, 0, 0, []
    with torch.inference_mode():
        for path in paired_paths(args.input_dir)[: args.limit]:
            reference_path = Path(args.reference_dir) / f"{path.stem}.png"
            if not reference_path.is_file():
                raise FileNotFoundError(f"Missing reference fused image: {reference_path}")
            visible, infrared = read_paired_image(path, args.image_size)
            reproduced = to_uint8(model(to_model_input(visible, device), to_model_input(infrared, device))).astype(np.int16)
            reference = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.int16)
            if reproduced.shape != reference.shape:
                raise ValueError(f"Shape mismatch for {path.name}: {reproduced.shape} vs {reference.shape}")
            delta = np.abs(reproduced - reference)
            maximum = max(maximum, int(delta.max()))
            total += int(delta.sum())
            pixels += delta.size
            checked.append(path.name)
    report = {"images_checked": checked, "max_absolute_uint8_difference": maximum, "mean_absolute_uint8_difference": total / pixels, "pass": maximum == 0}
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
