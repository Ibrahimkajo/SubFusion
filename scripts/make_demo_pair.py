"""Create a licence-free synthetic visible/infrared pair for a smoke test.

The generated PNG contains the visible RGB image in its left half and a
grayscale infrared-like image in its right half, matching the repository's
paired-input convention.  It is deliberately synthetic: it demonstrates the
I/O and inference path without redistributing any third-party dataset image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def build_pair(size: int) -> Image.Image:
    """Return a deterministic, horizontally paired RGB/IR smoke-test image."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    horizon = int(size * 0.48)

    # A coloured daytime-style visible scene with a road, buildings, and cars.
    visible = np.empty((size, size, 3), dtype=np.float32)
    sky = y < horizon
    visible[..., 0] = np.where(sky, 65 + 45 * y / max(horizon, 1), 62)
    visible[..., 1] = np.where(sky, 135 + 55 * y / max(horizon, 1), 66)
    visible[..., 2] = np.where(sky, 205 + 35 * y / max(horizon, 1), 72)
    visible[~sky] = (58, 61, 66)
    rgb = Image.fromarray(np.uint8(np.clip(visible, 0, 255)), "RGB")
    draw = ImageDraw.Draw(rgb)
    draw.rectangle((int(size * .06), int(size * .20), int(size * .29), horizon), fill=(84, 105, 125))
    draw.rectangle((int(size * .69), int(size * .10), int(size * .93), horizon), fill=(70, 89, 111))
    for left in (int(size * .10), int(size * .17), int(size * .74), int(size * .83)):
        draw.rectangle((left, int(size * .27), left + int(size * .045), int(size * .35)), fill=(195, 214, 225))
    draw.polygon([(0, size), (size, size), (int(size * .67), horizon), (int(size * .33), horizon)], fill=(54, 54, 57))
    draw.line((int(size * .43), horizon, int(size * .37), size), fill=(231, 203, 88), width=max(2, size // 90))
    draw.line((int(size * .57), horizon, int(size * .64), size), fill=(231, 203, 88), width=max(2, size // 90))
    draw.rounded_rectangle((int(size * .24), int(size * .59), int(size * .43), int(size * .71)), radius=size // 45, fill=(205, 58, 45))
    draw.rounded_rectangle((int(size * .57), int(size * .51), int(size * .78), int(size * .64)), radius=size // 45, fill=(37, 112, 197))
    for cx, cy in ((.29, .71), (.39, .71), (.62, .64), (.73, .64)):
        draw.ellipse((int(size * cx) - size // 45, int(size * cy) - size // 45, int(size * cx) + size // 45, int(size * cy) + size // 45), fill=(22, 22, 24))

    # Build a co-registered thermal-style image: dark background, bright cars.
    ir = np.full((size, size), 35.0, dtype=np.float32)
    ir += 18 * (y / max(size - 1, 1))
    ir[y < horizon] += 10
    ir_image = Image.fromarray(np.uint8(np.clip(ir, 0, 255)), "L")
    ir_draw = ImageDraw.Draw(ir_image)
    ir_draw.rounded_rectangle((int(size * .24), int(size * .59), int(size * .43), int(size * .71)), radius=size // 45, fill=238)
    ir_draw.rounded_rectangle((int(size * .57), int(size * .51), int(size * .78), int(size * .64)), radius=size // 45, fill=220)
    ir_draw.ellipse((int(size * .45), int(size * .40), int(size * .52), int(size * .62)), fill=250)
    ir_draw.ellipse((int(size * .79), int(size * .38), int(size * .85), int(size * .57)), fill=245)
    ir_image = ir_image.filter(ImageFilter.GaussianBlur(radius=max(0.6, size / 420)))
    infrared = Image.merge("RGB", (ir_image, ir_image, ir_image))

    pair = Image.new("RGB", (size * 2, size))
    pair.paste(rgb, (0, 0))
    pair.paste(infrared, (size, 0))
    return pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="examples/demo_input/synthetic_street.png")
    parser.add_argument("--size", type=int, default=480)
    args = parser.parse_args()
    if args.size < 64:
        raise ValueError("--size must be at least 64")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_pair(args.size).save(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
