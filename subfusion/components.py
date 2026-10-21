"""Mathematically explicit SVD components used by the SubFusion losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SVDComponents:
    """Complete rank-k and singular-vector geometry components of BCHW data."""

    luminance: torch.Tensor
    low_rank: torch.Tensor
    residual: torch.Tensor
    geometry: torch.Tensor
    rank: int


def rgb_to_luminance(image: torch.Tensor) -> torch.Tensor:
    """Return BT.601 luminance with shape ``[batch, height, width]``.

    The function is linear and therefore works for the project's normalised
    ``[-1, 1]`` network tensors. Infrared images are currently loaded as RGB;
    if their channels are replicated grayscale, this equals the IR channel.
    """
    if image.ndim != 4 or image.shape[1] < 3:
        raise ValueError(f"Expected BCHW image with at least three channels, got {tuple(image.shape)}")
    return 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]


def compose_visible_chroma(generated: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """Use generated luminance with visible Cb/Cr to form an RGB fused image.

    Both tensors use the project's normalised RGB domain.  The operation is
    differentiable and makes hue preservation an output constraint rather than
    a soft colour-loss preference.
    """
    if generated.shape != visible.shape or generated.ndim != 4 or generated.shape[1] < 3:
        raise ValueError("generated and visible must have matching BCHW RGB shapes")
    luminance = rgb_to_luminance(generated)
    visible_luminance = rgb_to_luminance(visible)
    cb = (visible[:, 2] - visible_luminance) * 0.564 + 0.5
    cr = (visible[:, 0] - visible_luminance) * 0.713 + 0.5
    blue = (cb - 0.5) / 0.564 + luminance
    red = (cr - 0.5) / 0.713 + luminance
    green = (luminance - 0.299 * red - 0.114 * blue) / 0.587
    return torch.stack((red, green, blue), dim=1)


def stable_svd(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the default fast CUDA SVD with exact convergence fallbacks."""
    if not matrix.is_cuda:
        return torch.linalg.svd(matrix, full_matrices=False)
    try:
        return torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError as exc:
        if "failed to converge" not in str(exc).lower():
            raise
        try:
            return torch.linalg.svd(matrix.double(), full_matrices=False, driver="gesvd")
        except RuntimeError as double_exc:
            if "failed to converge" not in str(double_exc).lower():
                raise
            cpu_u, cpu_singular_values, cpu_vh = torch.linalg.svd(
                matrix.to(device="cpu", dtype=torch.float64), full_matrices=False
            )
            return (
                cpu_u.to(device=matrix.device),
                cpu_singular_values.to(device=matrix.device),
                cpu_vh.to(device=matrix.device),
            )


def legacy_band_component(image: torch.Tensor, start: int = 1, stop: int = 255) -> torch.Tensor:
    """Reconstruct the historical per-RGB-channel SVD band ``[start:stop]``.

    This function exists solely to reproduce the pre-revision ``sparse``
    checkpoint family.  It deliberately retains the old resolution-dependent
    three-channel construction rather than presenting it as the mathematically
    complete residual ``X-L_k`` used by :func:`decompose_luminance`.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected BCHW image, got {tuple(image.shape)}")
    u, singular_values, vh = stable_svd(image)
    maximum_rank = singular_values.shape[-1]
    if not 0 <= start < stop <= maximum_rank:
        raise ValueError(
            f"Legacy SVD band must satisfy 0 <= start < stop <= {maximum_rank}; "
            f"got [{start}:{stop}]"
        )
    component = (u[..., start:stop] * singular_values[..., start:stop].unsqueeze(-2)) @ vh[..., start:stop, :]
    return component.to(dtype=image.dtype)


def decompose_luminance(luminance: torch.Tensor, rank: int) -> SVDComponents:
    """Compute ``L_k``, ``R_k`` and complete ``Q=UV^T`` for batched images.

    ``L_k = U_k Sigma_k V_k^T`` and ``R_k = X-L_k``.  ``Q`` intentionally
    uses *all* thin-SVD singular-vector pairs and contains no singular values.
    """
    if luminance.ndim != 3:
        raise ValueError(f"Expected [batch, height, width], got {tuple(luminance.shape)}")
    u, singular_values, vh = stable_svd(luminance)
    maximum_rank = singular_values.shape[-1]
    if not 1 <= rank <= maximum_rank:
        raise ValueError(f"rank must be in [1, {maximum_rank}], got {rank}")
    low_rank = (u[..., :rank] * singular_values[..., :rank].unsqueeze(-2)) @ vh[..., :rank, :]
    residual = luminance - low_rank
    geometry = u @ vh
    return SVDComponents(luminance, low_rank, residual, geometry, rank)


def top_fraction_mask(geometry: torch.Tensor, fraction: float = 0.10) -> torch.Tensor:
    """Return an exact, detached top-|Q| saliency mask per batch item."""
    if geometry.ndim != 3:
        raise ValueError(f"Expected [batch, height, width], got {tuple(geometry.shape)}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    batch, height, width = geometry.shape
    selected_count = max(1, int(round(fraction * height * width)))
    flat = geometry.detach().abs().reshape(batch, -1)
    indices = torch.topk(flat, k=selected_count, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask.reshape(batch, height, width)


def infrared_exclusive_residual_mask(
    visible_residual: torch.Tensor, infrared_residual: torch.Tensor, fraction: float = 0.10,
) -> torch.Tensor:
    """Detach a top-fraction mask for residual detail stronger in IR than visible.

    Scores are ``max(|R_ir|-|R_v|, 0)``.  The mask is source-only and
    therefore cannot be optimized by the fused branch.
    """
    if visible_residual.shape != infrared_residual.shape:
        raise ValueError('visible and infrared residuals must have the same shape')
    exclusive_strength = (infrared_residual.detach().abs() - visible_residual.detach().abs()).clamp_min(0)
    return top_fraction_mask(exclusive_strength, fraction)


def masked_geometry_l1(fused_geometry: torch.Tensor, source_geometry: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
    """Mean per-image L1 geometry error at detached salient source locations."""
    if fused_geometry.shape != source_geometry.shape or source_mask.shape != source_geometry.shape:
        raise ValueError("fused geometry, source geometry, and source mask must have the same shape")
    errors = (fused_geometry - source_geometry).abs() * source_mask.to(fused_geometry.dtype)
    return (errors.flatten(1).sum(dim=1) / source_mask.flatten(1).sum(dim=1).clamp_min(1)).mean()


def masked_component_l1(fused_component: torch.Tensor, target_component: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
    """Mean L1 component error at detached source-selected locations."""
    if fused_component.shape != target_component.shape or source_mask.shape != target_component.shape:
        raise ValueError('fused component, target component, and source mask must have the same shape')
    errors = (fused_component - target_component).abs() * source_mask.to(fused_component.dtype)
    return (errors.flatten(1).sum(dim=1) / source_mask.flatten(1).sum(dim=1).clamp_min(1)).mean()


def source_component_targets(
    visible: SVDComponents, infrared: SVDComponents, low_rank_rule: str = 'average',
    visible_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an explicit low-rank rule and the frozen summed residual rule."""
    if visible.rank != infrared.rank:
        raise ValueError("visible and infrared components must use the same rank")
    if low_rank_rule == 'average':
        low_rank = 0.5 * (visible.low_rank + infrared.low_rank)
    elif low_rank_rule == 'visible':
        low_rank = visible.low_rank
    elif low_rank_rule == 'weighted':
        if not 0.0 <= visible_weight <= 1.0:
            raise ValueError(f"visible_weight must be in [0, 1], got {visible_weight}")
        low_rank = visible_weight * visible.low_rank + (1.0 - visible_weight) * infrared.low_rank
    else:
        raise ValueError(f"Unsupported low-rank fusion rule: {low_rank_rule}")
    return low_rank, visible.residual + infrared.residual


def exposure_adaptive_low_rank_target(
    visible: SVDComponents,
    infrared: SVDComponents,
    min_visible_weight: float = 0.25,
    max_visible_weight: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a bounded, source-exposure-adaptive low-rank target.

    The visible coefficient is derived only from the fixed physical source
    luminances, ``mu_v / (mu_v + mu_ir)``, and is clipped to a declared
    interval so neither source can disappear from the low-rank target.  The
    source tensors use the project's ``[-1, 1]`` domain, hence they are
    converted to ``[0, 1]`` only for the exposure statistic.  The target keeps
    the SVD components themselves (and their singular values) unchanged.
    """
    if visible.rank != infrared.rank:
        raise ValueError("visible and infrared components must use the same rank")
    if not 0.0 <= min_visible_weight <= max_visible_weight <= 1.0:
        raise ValueError("visible-weight bounds must satisfy 0 <= min <= max <= 1")
    visible_mean = ((visible.luminance.detach() + 1.0) * 0.5).mean(dim=(-2, -1))
    infrared_mean = ((infrared.luminance.detach() + 1.0) * 0.5).mean(dim=(-2, -1))
    visible_weight = (visible_mean / (visible_mean + infrared_mean).clamp_min(1e-8)).clamp(
        min=min_visible_weight, max=max_visible_weight,
    )
    target = (
        visible_weight[:, None, None] * visible.low_rank
        + (1.0 - visible_weight)[:, None, None] * infrared.low_rank
    )
    return target, visible_weight
