#!/usr/bin/env python3
"""Directional luminance-correction constraints for rolling-shutter flicker.

The standard Restormer is treated as an estimator of a luminance correction
field.  Instead of accepting all spatial structure in that field, we retain
primarily the row-varying component expected from rolling-shutter flicker.

The default representation is a stabilized log-gain:

    g = log((Y_pred + eps) / (Y_input + eps))

which is horizontally low-pass filtered before being applied to the original
full-resolution luminance.  The small positive eps stabilizes deep shadows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CorrectionStats:
    raw_rms: float
    constrained_rms: float
    removed_rms: float
    raw_mean: float
    constrained_mean: float


def _gaussian_kernel1d(sigma: float, radius: int, *, device, dtype) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum()


def _smooth_axis(x: torch.Tensor, sigma: float, axis: str) -> torch.Tensor:
    if sigma <= 0:
        return x
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected Bx1xHxW correction tensor, got {tuple(x.shape)}")
    if axis == "x":
        n = x.shape[-1]
    elif axis == "y":
        n = x.shape[-2]
    else:
        raise ValueError("axis must be 'x' or 'y'")
    if n < 2:
        return x
    radius = min(max(1, int(math.ceil(3.0 * sigma))), n - 1)
    kernel = _gaussian_kernel1d(sigma, radius, device=x.device, dtype=x.dtype)
    if axis == "x":
        padded = F.pad(x, (radius, radius, 0, 0), mode="reflect")
        return F.conv2d(padded, kernel.view(1, 1, 1, -1))
    padded = F.pad(x, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel.view(1, 1, -1, 1))


def _preserve_row_mean(raw: torch.Tensor, filtered: torch.Tensor) -> torch.Tensor:
    """Keep each row's average correction equal to the Restormer estimate."""
    return filtered + raw.mean(dim=-1, keepdim=True) - filtered.mean(dim=-1, keepdim=True)


def _luma_confidence(y: torch.Tensor) -> torch.Tensor:
    """Downweight near-black and near-clipped pixels for strict row profiles."""
    low = ((y - 0.01) / 0.08).clamp(0.0, 1.0)
    high = ((0.995 - y) / 0.10).clamp(0.0, 1.0)
    return low * high + 1e-4


def make_correction_field(
    input_y: torch.Tensor,
    predicted_y: torch.Tensor,
    *,
    mode: str = "directional",
    horizontal_sigma: float = 16.0,
    vertical_sigma: float = 0.0,
    eps: float = 0.02,
    clip_stops: float = 2.0,
    row_anchor: bool = True,
) -> tuple[str, torch.Tensor, CorrectionStats]:
    """Build a constrained correction field at the model working resolution.

    Returns ``(domain, field, stats)``.  ``domain`` is either ``additive`` or
    ``log`` and determines how :func:`apply_correction_field` interprets it.

    Modes:
      raw                  Exact legacy/v2 additive Restormer correction.
      directional          Stabilized log-gain, horizontally smoothed.
      directional-additive Additive delta-Y, horizontally smoothed.
      row                  One robust weighted correction value per row.
    """
    if input_y.shape != predicted_y.shape or input_y.ndim != 4 or input_y.shape[1] != 1:
        raise ValueError(f"Expected matching Bx1xHxW Y tensors, got {tuple(input_y.shape)} and {tuple(predicted_y.shape)}")
    if horizontal_sigma < 0 or vertical_sigma < 0:
        raise ValueError("Smoothing sigmas must be >= 0")
    if eps < 0:
        raise ValueError("eps must be >= 0")
    if clip_stops <= 0:
        raise ValueError("clip_stops must be > 0")

    mode = mode.lower()
    if mode == "raw":
        raw = predicted_y - input_y
        constrained = raw
        domain = "additive"
    elif mode == "directional-additive":
        raw = predicted_y - input_y
        constrained = _smooth_axis(raw, horizontal_sigma, "x")
        if row_anchor:
            constrained = _preserve_row_mean(raw, constrained)
        constrained = _smooth_axis(constrained, vertical_sigma, "y")
        domain = "additive"
    else:
        safe_in = (input_y.float() + eps).clamp_min(1e-6)
        safe_pred = (predicted_y.float() + eps).clamp_min(1e-6)
        raw = torch.log(safe_pred) - torch.log(safe_in)
        limit = float(clip_stops) * math.log(2.0)
        raw = raw.clamp(-limit, limit)
        domain = "log"
        if mode == "directional":
            constrained = _smooth_axis(raw, horizontal_sigma, "x")
            if row_anchor:
                constrained = _preserve_row_mean(raw, constrained)
            constrained = _smooth_axis(constrained, vertical_sigma, "y")
        elif mode == "row":
            w = _luma_confidence(input_y.float())
            row = (raw * w).sum(dim=-1, keepdim=True) / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            constrained = row.expand_as(raw)
            constrained = _smooth_axis(constrained, vertical_sigma, "y")
        else:
            raise ValueError(f"Unsupported correction mode: {mode}")

    with torch.no_grad():
        rawf = raw.float()
        conf = constrained.float()
        stats = CorrectionStats(
            raw_rms=float(rawf.square().mean().sqrt()),
            constrained_rms=float(conf.square().mean().sqrt()),
            removed_rms=float((rawf - conf).square().mean().sqrt()),
            raw_mean=float(rawf.mean()),
            constrained_mean=float(conf.mean()),
        )
    return domain, constrained, stats


def apply_correction_field(
    original_y: torch.Tensor,
    field: torch.Tensor,
    *,
    domain: str,
    eps: float = 0.02,
    strength: float = 1.0,
) -> torch.Tensor:
    """Resize and apply a model-resolution correction to full-resolution Y."""
    if original_y.ndim != 4 or original_y.shape[1] != 1:
        raise ValueError(f"Expected Bx1xHxW original Y, got {tuple(original_y.shape)}")
    resized = F.interpolate(field.float(), size=original_y.shape[-2:], mode="bilinear", align_corners=False, antialias=True)
    if domain == "additive":
        return original_y.float() + resized * float(strength)
    if domain == "log":
        gain = torch.exp(resized * float(strength))
        return (original_y.float() + float(eps)) * gain - float(eps)
    raise ValueError(f"Unsupported correction domain: {domain}")


def remove_global_dc(
    field: torch.Tensor,
    reference_y: torch.Tensor,
    *,
    domain: str,
) -> tuple[torch.Tensor, float]:
    """Remove the weighted global DC component of a correction field.

    Intended mainly for pass 2. It preserves local row/band structure while
    preventing a recursive pass from accumulating a global exposure shift.
    Returns ``(field_without_dc, removed_shift_stops)``. For additive fields,
    the reported stops value is only an approximate diagnostic around mid-gray.
    """
    if field.ndim != 4 or field.shape[1] != 1:
        raise ValueError(f"Expected Bx1xHxW field, got {tuple(field.shape)}")
    if reference_y.ndim != 4 or reference_y.shape[1] != 1:
        raise ValueError(f"Expected Bx1xHxW reference Y, got {tuple(reference_y.shape)}")
    ref = F.interpolate(reference_y.float(), size=field.shape[-2:], mode="bilinear", align_corners=False, antialias=True)
    w = _luma_confidence(ref)
    dc = (field.float() * w).sum(dim=(-2, -1), keepdim=True) / w.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    out = field.float() - dc
    if domain == "log":
        stops = float(dc.mean() / math.log(2.0))
    elif domain == "additive":
        # Approximate exposure-equivalent shift for logging only.
        stops = float(torch.log2((0.5 + dc.mean()).clamp_min(1e-6) / 0.5))
    else:
        raise ValueError(f"Unsupported correction domain: {domain}")
    return out, stops
