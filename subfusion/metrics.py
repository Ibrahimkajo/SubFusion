"""Python image-fusion metrics used by :mod:`evaluate_checkpoints`.

All source images are converted to an 8-bit luminance plane before the
source/fusion metrics are evaluated.  DeltaE is intentionally different: it
is measured on RGB fused and visible images in CIELAB.

Implementation references
-------------------------
* EN, SF, AG, MI, CC, SCD, Qabf and VIF follow the public Python evaluator
  released with MMIF-CDDFuse:
  https://github.com/Zhaozixiang1228/MMIF-CDDFuse/blob/main/utils/Evaluator.py
* Qy follows Yang et al., "A Novel Similarity Based Quality Metric for Image
  Fusion", Information Fusion 9(2), 2008.  The public MFIF reference code is
  https://github.com/yuliu316316/MFIF/blob/master/evaluation/objective_metrics/metricYang.m
  This is an independent NumPy/SciPy implementation, not a MATLAB call.
* MS-SSIM uses the public ``pytorch-msssim`` implementation:
  https://github.com/VainF/pytorch-msssim
* FMI_w uses a wavelet-feature, normalized-mutual-information formulation.
  Its transform/normalisation settings are recorded in every output JSON so
  values are reproducible and are not silently mixed with another FMI variant.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
from pytorch_msssim import ms_ssim
from scipy.ndimage import convolve
from scipy.signal import convolve2d
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

EPS = np.finfo(np.float64).eps


def to_gray_uint8(image: np.ndarray) -> np.ndarray:
    """Return a two-dimensional uint8 luminance image.

    Inputs produced by this project are RGB.  The fixed BT.601 conversion
    avoids comparing one source in RGB and another in grayscale by accident.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        return np.clip(np.rint(image), 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected grayscale or RGB image, got {image.shape}")
    rgb = image[..., :3].astype(np.float64)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def entropy(image: np.ndarray) -> float:
    hist = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def spatial_frequency(image: np.ndarray) -> float:
    image = image.astype(np.float64)
    rf = np.mean(np.diff(image, axis=1) ** 2)
    cf = np.mean(np.diff(image, axis=0) ** 2)
    return float(np.sqrt(rf + cf))


def average_gradient(image: np.ndarray) -> float:
    image = image.astype(np.float64)
    gx = np.empty_like(image)
    gy = np.empty_like(image)
    gx[:, 0] = image[:, 1] - image[:, 0]
    gx[:, -1] = image[:, -1] - image[:, -2]
    gx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2.0
    gy[0, :] = image[1, :] - image[0, :]
    gy[-1, :] = image[-1, :] - image[-2, :]
    gy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2.0
    return float(np.mean(np.sqrt((gx**2 + gy**2) / 2.0)))


def _mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    joint = np.histogram2d(a.ravel(), b.ravel(), bins=256, range=((0, 256), (0, 256)))[0]
    joint /= joint.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log(joint[nz] / (pa * pb)[nz])))


def mutual_information(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    return _mutual_information(fused, visible) + _mutual_information(fused, infrared)


def correlation_coefficient(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    return 0.5 * (_correlation(visible, fused) + _correlation(infrared, fused))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float64).ravel()
    second = second.astype(np.float64).ravel()
    denominator = np.linalg.norm(first - first.mean()) * np.linalg.norm(second - second.mean())
    return float(np.dot(first - first.mean(), second - second.mean()) / denominator) if denominator else 0.0


def scd(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    return _correlation(visible, fused - infrared) + _correlation(infrared, fused - visible)


def _sobel_strength_angle(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    h_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    x = convolve(image.astype(np.float64), h_x, mode="constant", cval=0.0)
    y = convolve(image.astype(np.float64), h_y, mode="constant", cval=0.0)
    angle = np.full_like(x, math.pi / 2)
    np.arctan(np.divide(y, x, out=np.zeros_like(y), where=x != 0), out=angle, where=x != 0)
    return np.hypot(x, y), angle


def _qabf_pair(source_g: np.ndarray, source_a: np.ndarray, fused_g: np.ndarray, fused_a: np.ndarray) -> np.ndarray:
    ratio = np.divide(np.minimum(source_g, fused_g), np.maximum(source_g, fused_g), out=np.zeros_like(source_g), where=np.maximum(source_g, fused_g) > 0)
    ratio[(source_g == fused_g) & (source_g > 0)] = 1.0
    angle = 1.0 - np.abs(source_a - fused_a) / (math.pi / 2.0)
    qg = 0.9994 / (1.0 + np.exp(-15.0 * (ratio - 0.5)))
    qa = 0.9879 / (1.0 + np.exp(-22.0 * (angle - 0.8)))
    return qg * qa


def qabf(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    vg, va = _sobel_strength_angle(visible)
    ig, ia = _sobel_strength_angle(infrared)
    fg, fa = _sobel_strength_angle(fused)
    denominator = np.sum(vg + ig)
    if denominator <= EPS:
        return 0.0
    return float(np.sum(_qabf_pair(vg, va, fg, fa) * vg + _qabf_pair(ig, ia, fg, fa) * ig) / denominator)


def _vif_pair(reference: np.ndarray, distorted: np.ndarray) -> float:
    """Four-scale pixel-domain VIF from the public MMIF-CDDFuse evaluator."""
    reference = reference.astype(np.float64)
    distorted = distorted.astype(np.float64)
    numerator = denominator = 0.0
    for scale in range(1, 5):
        size = 2 ** (5 - scale) + 1
        sigma = size / 5.0
        y, x = np.ogrid[-(size - 1) / 2:(size + 1) / 2, -(size - 1) / 2:(size + 1) / 2]
        window = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        window /= window.sum()
        if scale > 1:
            reference = convolve2d(reference, window[::-1, ::-1], mode="valid")[::2, ::2]
            distorted = convolve2d(distorted, window[::-1, ::-1], mode="valid")[::2, ::2]
        mu_r = convolve2d(reference, window[::-1, ::-1], mode="valid")
        mu_d = convolve2d(distorted, window[::-1, ::-1], mode="valid")
        var_r = np.maximum(convolve2d(reference * reference, window[::-1, ::-1], mode="valid") - mu_r**2, 0)
        var_d = np.maximum(convolve2d(distorted * distorted, window[::-1, ::-1], mode="valid") - mu_d**2, 0)
        cov = convolve2d(reference * distorted, window[::-1, ::-1], mode="valid") - mu_r * mu_d
        gain = cov / (var_r + 1e-10)
        noise = var_d - gain * cov
        gain[var_r < 1e-10] = 0
        noise[var_r < 1e-10] = var_d[var_r < 1e-10]
        gain[var_d < 1e-10] = 0
        noise[var_d < 1e-10] = 0
        noise[gain < 0] = var_d[gain < 0]
        gain[gain < 0] = 0
        noise = np.maximum(noise, 1e-10)
        numerator += np.log10(1 + gain**2 * var_r / (noise + 2)).sum()
        denominator += np.log10(1 + var_r / 2).sum()
    return float(numerator / denominator) if denominator > EPS else 1.0


def vif(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    return _vif_pair(visible, fused) + _vif_pair(infrared, fused)


def _yang_ssim_maps(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.arange(-3, 4, dtype=np.float64)
    gaussian = np.exp(-(axis**2) / (2.0 * 1.5**2))
    window = np.outer(gaussian, gaussian)
    window /= window.sum()
    first = first.astype(np.float64)
    second = second.astype(np.float64)
    mu_first = convolve2d(first, window, mode="valid")
    mu_second = convolve2d(second, window, mode="valid")
    var_first = convolve2d(first * first, window, mode="valid") - mu_first**2
    var_second = convolve2d(second * second, window, mode="valid") - mu_second**2
    covariance = convolve2d(first * second, window, mode="valid") - mu_first * mu_second
    score = ((2 * mu_first * mu_second + 2e-16) * (2 * covariance + 2e-16)) / ((mu_first**2 + mu_second**2 + 2e-16) * (var_first + var_second + 2e-16))
    return score, var_first, var_second


def qy(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    similarity_sources, variance_visible, variance_infrared = _yang_ssim_maps(visible, infrared)
    similarity_visible, _, _ = _yang_ssim_maps(visible, fused)
    similarity_infrared, _, _ = _yang_ssim_maps(infrared, fused)
    total_variance = variance_visible + variance_infrared
    visible_weight = np.divide(variance_visible, total_variance, out=np.full_like(total_variance, 0.5), where=total_variance > EPS)
    redundant = similarity_sources >= 0.75
    values = np.where(redundant, visible_weight * similarity_visible + (1.0 - visible_weight) * similarity_infrared, np.maximum(similarity_visible, similarity_infrared))
    return float(values.mean())


def _haar_level(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One pure-NumPy, orthonormal 2D Haar decomposition level."""
    image = image[: image.shape[0] - image.shape[0] % 2, : image.shape[1] - image.shape[1] % 2]
    top_left = image[0::2, 0::2]
    top_right = image[0::2, 1::2]
    bottom_left = image[1::2, 0::2]
    bottom_right = image[1::2, 1::2]
    low_low = (top_left + top_right + bottom_left + bottom_right) / 2.0
    low_high = (top_left - top_right + bottom_left - bottom_right) / 2.0
    high_low = (top_left + top_right - bottom_left - bottom_right) / 2.0
    high_high = (top_left - top_right - bottom_left + bottom_right) / 2.0
    return low_low, low_high, high_low, high_high


def _wavelet_features(image: np.ndarray) -> np.ndarray:
    """Level-two Haar subbands without a compiled wavelet-package dependency."""
    low_low_1, low_high_1, high_low_1, high_high_1 = _haar_level(image.astype(np.float64))
    low_low_2, low_high_2, high_low_2, high_high_2 = _haar_level(low_low_1)
    return np.concatenate((low_low_2.ravel(), low_high_2.ravel(), high_low_2.ravel(), high_high_2.ravel(), low_high_1.ravel(), high_low_1.ravel(), high_high_1.ravel()))


def _normalised_mi(a: np.ndarray, b: np.ndarray) -> float:
    # Quantile binning makes feature distributions comparable across modalities.
    edges_a = np.quantile(a, np.linspace(0, 1, 65))
    edges_b = np.quantile(b, np.linspace(0, 1, 65))
    a_q = np.clip(np.searchsorted(np.unique(edges_a), a, side="right") - 1, 0, 63).astype(np.uint8)
    b_q = np.clip(np.searchsorted(np.unique(edges_b), b, side="right") - 1, 0, 63).astype(np.uint8)
    mi = _mutual_information(a_q * 4, b_q * 4)
    ha = entropy(a_q * 4)
    hb = entropy(b_q * 4)
    return 2.0 * mi / (ha + hb) if ha + hb > EPS else 1.0


def fmi_w(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    """Wavelet-domain feature mutual information (symmetric source average)."""
    fused_f = _wavelet_features(fused)
    return 0.5 * (_normalised_mi(_wavelet_features(visible), fused_f) + _normalised_mi(_wavelet_features(infrared), fused_f))


def fused_ssim(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    return float(0.5 * (structural_similarity(fused, visible, data_range=255) + structural_similarity(fused, infrared, data_range=255)))


def fused_ms_ssim(fused: np.ndarray, visible: np.ndarray, infrared: np.ndarray) -> float:
    def calculate(reference: np.ndarray) -> float:
        source = torch.from_numpy(reference.astype(np.float32))[None, None]
        result = torch.from_numpy(fused.astype(np.float32))[None, None]
        return float(ms_ssim(source, result, data_range=255, size_average=True))
    return 0.5 * (calculate(visible) + calculate(infrared))


def delta_e00(fused_rgb: np.ndarray, visible_rgb: np.ndarray) -> float:
    fused_lab = rgb2lab(fused_rgb[..., :3].astype(np.float64) / 255.0)
    visible_lab = rgb2lab(visible_rgb[..., :3].astype(np.float64) / 255.0)
    return float(deltaE_ciede2000(fused_lab, visible_lab).mean())


def calculate_metrics(fused_rgb: np.ndarray, visible_rgb: np.ndarray, infrared_rgb: np.ndarray) -> Dict[str, float]:
    fused = to_gray_uint8(fused_rgb)
    visible = to_gray_uint8(visible_rgb)
    infrared = to_gray_uint8(infrared_rgb)
    if fused.shape != visible.shape or fused.shape != infrared.shape:
        raise ValueError("fused, visible, and infrared images must have equal spatial dimensions")
    return {
        "EN": entropy(fused),
        "SF": spatial_frequency(fused),
        "AG": average_gradient(fused),
        "MI": mutual_information(fused, visible, infrared),
        "CC": correlation_coefficient(fused, visible, infrared),
        "SCD": scd(fused, visible, infrared),
        "Qabf": qabf(fused, visible, infrared),
        "VIF": vif(fused, visible, infrared),
        "Qy": qy(fused, visible, infrared),
        "SSIM": fused_ssim(fused, visible, infrared),
        "MS_SSIM": fused_ms_ssim(fused, visible, infrared),
        "FMI_w": fmi_w(fused, visible, infrared),
        "DeltaE00": delta_e00(fused_rgb, visible_rgb),
    }
