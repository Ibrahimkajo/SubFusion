"""Train the generator-only SubFusion U-Net using a YAML configuration."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from subfusion.data import PairedFusionDataset
from subfusion.losses import FullLossSettings, FullSubFusionLoss
from subfusion.model import SubFusionUNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "msrs_full_rank1_70e.yaml"))
    parser.add_argument("--data-dir", required=True, help="Folder of horizontally paired visible-left/IR-right training files.")
    parser.add_argument("--output-dir", default="runs/msrs_full_rank1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    device = torch.device(args.device)
    dataset = PairedFusionDataset(args.data_dir, config["load_size"], config["crop_size"], config["horizontal_flip"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=device.type == "cuda")
    model = SubFusionUNet().to(device)
    criterion = FullSubFusionLoss(FullLossSettings(**config["loss"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), betas=(float(config["beta1"]), 0.999))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        totals: dict[str, float] = {}
        for visible, infrared in loader:
            visible, infrared = visible.to(device), infrared.to(device)
            optimizer.zero_grad(set_to_none=True)
            total, terms = criterion(model(visible, infrared), visible, infrared)
            total.backward()
            optimizer.step()
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + float(value)
        row = {"epoch": epoch, **{name: value / len(loader) for name, value in totals.items()}}
        history.append(row)
        print(row)
        torch.save(model.state_dict(), output / "latest_net_G.pth")

    with (output / "loss_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    main()
