# SubFusion: a two-input U-Net for visible–infrared image fusion

This repository is a self-contained reviewer release of the SubFusion fusion network.  It contains a two-input U-Net, direct image-domain and subspace losses, reproducible inference/evaluation scripts, and the final MSRS-trained checkpoint.

The implementation contains one fusion network only.  Visible RGB and infrared inputs are processed by separate encoders, merged by learned 1×1 feature-fusion layers at each U-Net scale, and reconstructed by one shared decoder.

## Included checkpoint

`weights/subfusion_msrs_full_rank1_70e.pth` is the final MSRS-trained checkpoint used for the reported 70-epoch result.

| Item | Value |
|---|---|
| Architecture | Two-input U-Net, 3-channel visible + replicated-RGB IR → 3-channel fused RGB |
| Training data | MSRS training split, 1,083 paired images |
| Training crop | 480 × 480 |
| Epochs / batch size | 70 / 4 |
| Optimizer | Adam, learning rate 1e-4, β1 = 0.5 |
| Random seed | 20260923 |
| SHA-256 | `A2C5BE010AADB4F1E496515252C201B1F46B051D6B87006C9F2B327A2A62DAD7` |

The checkpoint file is 146 MB.  GitHub rejects ordinary files larger than 100 MB, so it is published as a GitHub Release asset.
## Full loss used for this checkpoint

All image tensors are normalized to `[-1, 1]`.  The final MSRS checkpoint uses

\[
\mathcal{L}=100\mathcal{L}_{A}+100\mathcal{L}_{B}+2\mathcal{L}_{R}+\mathcal{L}_{S}+\mathcal{L}_{C}.
\]

`L_A` and `L_B` are RGB L1 terms with visible and replicated-RGB infrared sources, respectively.  `L_R` is the complete per-RGB rank-1 residual target, included to reproduce the released checkpoint exactly.  `L_C` preserves visible chromatic components.  `L_S` compares the complete singular-vector geometry map \(Q=UV^T\) at the top 10% of source-geometry locations.
The exact configuration is in `configs/msrs_full_rank1_70e.yaml`.

## Installation

```bash
conda create -n subfusion python=3.10 -y
conda activate subfusion
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA system from [pytorch.org](https://pytorch.org/get-started/locally/) if the generic `torch` package does not provide the desired GPU build.

## Data layout

The scripts expect horizontally paired image files:

```text
paired_images/
├── 00001.png    # left half = visible RGB; right half = infrared
├── 00002.png
└── ...
```

Images are resized to 480 × 480 independently for inference/evaluation.  Datasets are not redistributed here; please obtain MSRS, LLVIP, or M3FD from their official sources and follow their terms of use.

## Inference

```bash
python scripts/infer.py \
  --input-dir /path/to/paired_test_images \
  --output-dir outputs/msrs_checkpoint
```

The script writes one RGB fused PNG per paired input image.

### One-image smoke test

The release includes one paired LLVIP input at
`examples/demo_input/llvip_010001.jpg`, so reviewers can test inference
immediately after downloading the checkpoint:

```bash
python scripts/infer.py --input-dir examples/demo_input --output-dir outputs/demo
```

This writes `outputs/demo/llvip_010001.png`.  The example is included only to
verify installation, paired-image I/O, checkpoint loading, and fusion
execution; it is not used for a quantitative claim.  It remains subject to the
official LLVIP non-commercial terms and required citation; see
`examples/README.md`.

## Evaluation

```bash
python scripts/evaluate.py \
  --input-dir /path/to/paired_test_images \
  --output-dir outputs/evaluation
```

This creates fused images, per-image values, and `summary_metrics.csv`.  The project metrics include EN, SF, AG, MI, CC, SCD, Qabf, VIF, Qy, SSIM, MS-SSIM, FMI_w, and DeltaE00.  The public IVIF-ZOO implementation remains an external dependency and is documented separately in the paper evaluation protocol.

## Training

```bash
python scripts/train.py \
  --config configs/msrs_full_rank1_70e.yaml \
  --data-dir /path/to/msrs_train_paired \
  --output-dir runs/msrs_full_rank1
```

Training produces a generator checkpoint and a CSV loss log.  GPU execution is recommended because the structure term calculates batched SVDs.


## Repository contents

* `subfusion/` — U-Net architecture, SVD components, losses, paired-image I/O, and project metrics.
* `scripts/` — training, inference, and evaluation.
* `configs/` — final MSRS configuration.
* `weights/` — final MSRS checkpoint and checksum information.
