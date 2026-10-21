"""The paired reconstruction and subspace losses used for SubFusion training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .components import decompose_luminance, legacy_band_component, masked_geometry_l1, rgb_to_luminance, top_fraction_mask


@dataclass(frozen=True)
class FullLossSettings:
    lambda_l1: float = 100.0
    visible_l1: float = 1.0
    infrared_l1: float = 1.0
    residual: float = 2.0
    structure: float = 1.0
    colour: float = 1.0
    rank: int = 1
    salient_fraction: float = 0.10
    visible_structure_weight: float = 0.5
    infrared_structure_weight: float = 0.5


class FullSubFusionLoss(nn.Module):
    """Exact loss family of the distributed MSRS full-loss checkpoint.

    This checkpoint used RGB image-domain L1 terms, a complete per-RGB rank-1
    residual target, visible-chroma fidelity, and the top-10% singular-vector
    geometry term.  All terms are optimized directly through one U-Net.
    """

    def __init__(self, settings: FullLossSettings = FullLossSettings()):
        super().__init__()
        self.settings = settings
        self.l1 = nn.L1Loss()

    def forward(self, fused: torch.Tensor, visible: torch.Tensor, infrared: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.settings
        visible_l1 = self.l1(fused, visible) * cfg.lambda_l1 * cfg.visible_l1
        infrared_l1 = self.l1(fused, infrared) * cfg.lambda_l1 * cfg.infrared_l1

        residual_visible = legacy_band_component(visible, start=1, stop=min(visible.shape[-2:]))
        residual_infrared = legacy_band_component(infrared, start=1, stop=min(infrared.shape[-2:]))
        residual_fused = legacy_band_component(fused, start=1, stop=min(fused.shape[-2:]))
        residual = self.l1(residual_fused, residual_visible + residual_infrared) * cfg.lambda_l1

        y_visible = rgb_to_luminance(visible)
        y_fused = rgb_to_luminance(fused)
        cb_visible = (visible[:, 2] - y_visible) * 0.564 + 0.5
        cr_visible = (visible[:, 0] - y_visible) * 0.713 + 0.5
        cb_fused = (fused[:, 2] - y_fused) * 0.564 + 0.5
        cr_fused = (fused[:, 0] - y_fused) * 0.713 + 0.5
        colour = (self.l1(cb_visible, cb_fused) + self.l1(cr_visible, cr_fused)) * cfg.lambda_l1

        visible_components = decompose_luminance(y_visible, cfg.rank)
        infrared_components = decompose_luminance(rgb_to_luminance(infrared), cfg.rank)
        fused_components = decompose_luminance(y_fused, cfg.rank)
        visible_mask = top_fraction_mask(visible_components.geometry, cfg.salient_fraction)
        infrared_mask = top_fraction_mask(infrared_components.geometry, cfg.salient_fraction)
        structure_visible = masked_geometry_l1(fused_components.geometry, visible_components.geometry, visible_mask) * cfg.lambda_l1
        structure_infrared = masked_geometry_l1(fused_components.geometry, infrared_components.geometry, infrared_mask) * cfg.lambda_l1
        structure = cfg.visible_structure_weight * structure_visible + cfg.infrared_structure_weight * structure_infrared

        total = visible_l1 + infrared_l1 + cfg.residual * residual + cfg.colour * colour + cfg.structure * structure
        return total, {
            "total": total.detach(), "visible_l1": visible_l1.detach(), "infrared_l1": infrared_l1.detach(),
            "residual": residual.detach(), "colour": colour.detach(), "structure": structure.detach(),
        }
