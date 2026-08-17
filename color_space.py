#!/usr/bin/env python3
"""Differentiable full-range BT.601-style YCbCr helpers.

Cb/Cr are represented zero-centered (neutral gray = 0) because that makes
chroma corrections and a 2-channel network head natural.
"""
from __future__ import annotations

import torch


def _check_rgb(x: torch.Tensor) -> None:
    if x.ndim not in (3, 4) or x.shape[-3] != 3:
        raise ValueError(f"Expected RGB tensor (...,3,H,W), got {tuple(x.shape)}")


def rgb_to_y_cbcr(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Y and centered CbCr with the same leading dimensions as input."""
    _check_rgb(x)
    r = x[..., 0:1, :, :]
    g = x[..., 1:2, :, :]
    b = x[..., 2:3, :, :]
    y = 0.299000 * r + 0.587000 * g + 0.114000 * b
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b
    cr = 0.500000 * r - 0.418688 * g - 0.081312 * b
    return y, torch.cat([cb, cr], dim=-3)


def y_cbcr_to_rgb(y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`rgb_to_y_cbcr` for centered Cb/Cr."""
    if y.ndim not in (3, 4) or y.shape[-3] != 1:
        raise ValueError(f"Expected Y tensor (...,1,H,W), got {tuple(y.shape)}")
    if cbcr.ndim != y.ndim or cbcr.shape[-3] != 2 or cbcr.shape[:-3] != y.shape[:-3] or cbcr.shape[-2:] != y.shape[-2:]:
        raise ValueError(f"Incompatible CbCr tensor: Y={tuple(y.shape)} CbCr={tuple(cbcr.shape)}")
    cb = cbcr[..., 0:1, :, :]
    cr = cbcr[..., 1:2, :, :]
    r = y + 1.402000 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772000 * cb
    return torch.cat([r, g, b], dim=-3)


def rgb_to_cbcr(x: torch.Tensor) -> torch.Tensor:
    return rgb_to_y_cbcr(x)[1]


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    return rgb_to_y_cbcr(x)[0]


def clamp_y_for_cbcr(y: torch.Tensor, cbcr: torch.Tensor) -> torch.Tensor:
    """Clamp Y to the RGB gamut while preserving centered Cb/Cr exactly."""
    if y.ndim not in (3, 4) or y.shape[-3] != 1:
        raise ValueError(f"Expected Y tensor (...,1,H,W), got {tuple(y.shape)}")
    if cbcr.ndim != y.ndim or cbcr.shape[-3] != 2:
        raise ValueError("CbCr must match Y and have two channels")
    cb = cbcr[..., 0:1, :, :]
    cr = cbcr[..., 1:2, :, :]
    dr = 1.402000 * cr
    dg = -0.344136 * cb - 0.714136 * cr
    db = 1.772000 * cb
    d = torch.cat([dr, dg, db], dim=-3)
    lower = (-d).amax(dim=-3, keepdim=True)
    upper = (1.0 - d).amin(dim=-3, keepdim=True)
    # Original in-gamut colors should have lower<=upper; guard numerical edge cases.
    mid = 0.5 * (lower + upper)
    lower = torch.where(lower <= upper, lower, mid)
    upper = torch.where(lower <= upper, upper, mid)
    return torch.maximum(torch.minimum(y, upper), lower)


def y_cbcr_to_rgb_preserve_y(y: torch.Tensor, cbcr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert to in-gamut RGB by scaling chroma only, preserving Y.

    Returns (rgb, alpha), where alpha in [0,1] is the per-pixel chroma scale.
    alpha=1 means the requested chroma was already in gamut.
    """
    if y.ndim not in (3, 4) or y.shape[-3] != 1:
        raise ValueError(f"Expected Y tensor (...,1,H,W), got {tuple(y.shape)}")
    if cbcr.ndim != y.ndim or cbcr.shape[-3] != 2:
        raise ValueError("CbCr must match Y and have two channels")
    y_safe = y.clamp(0.0, 1.0)
    cb = cbcr[..., 0:1, :, :]
    cr = cbcr[..., 1:2, :, :]
    d = torch.cat([
        1.402000 * cr,
        -0.344136 * cb - 0.714136 * cr,
        1.772000 * cb,
    ], dim=-3)
    inf = torch.full_like(d, float("inf"))
    pos_limit = torch.where(d > 1e-12, (1.0 - y_safe) / d, inf)
    neg_limit = torch.where(d < -1e-12, y_safe / (-d), inf)
    limit = torch.minimum(pos_limit, neg_limit).amin(dim=-3, keepdim=True)
    alpha = torch.minimum(torch.ones_like(limit), limit).clamp(0.0, 1.0)
    rgb = y_safe + alpha * d
    return rgb.clamp(0.0, 1.0), alpha
