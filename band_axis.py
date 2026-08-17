#!/usr/bin/env python3
"""Band-axis detection and orientation normalization for row-trained restoration.

The Restormer branches and deterministic post-filters were trained/designed for
bands that vary along image Y (visible horizontal stripes).  A camera rotated
90 degrees produces the same sensor-row artifact as visible vertical stripes
once the photograph is displayed upright.  This module normalizes such cases
by rotating them into the row-oriented processing domain and rotating the
result back afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from color_space import rgb_to_y_cbcr


@dataclass(frozen=True)
class BandAxisDecision:
    axis: str
    reason: str
    horizontal_score: float
    vertical_score: float
    aspect_ratio: float

    @property
    def needs_rotation(self) -> bool:
        return self.axis == "vertical"


def _poly_detrended_rms(profile: torch.Tensor, degree: int = 2) -> float:
    """Low-frequency RMS after removing a small polynomial baseline."""
    p = profile.detach().float().flatten()
    n = int(p.numel())
    if n < 12:
        return 0.0
    t = torch.linspace(-1.0, 1.0, n, device=p.device, dtype=p.dtype)
    cols = [torch.ones_like(t)]
    for k in range(1, int(degree) + 1):
        cols.append(t.pow(k))
    X = torch.stack(cols, dim=1)
    # Tiny ridge is more deterministic than depending on lstsq driver details.
    eye = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    coef = torch.linalg.solve(X.T @ X + 1e-6 * eye, X.T @ p)
    r = p - X @ coef
    # Suppress pixel/noise-scale texture; orientation selection is about broad
    # coherent stripes, not fine sensor noise.
    if n >= 9:
        k = 7
        r = F.avg_pool1d(r.view(1, 1, -1), kernel_size=k, stride=1, padding=k // 2).view(-1)
    r = r - r.median()
    mad = (r - r.median()).abs().median() * 1.4826 + 1e-6
    r = r.clamp(-4.0 * mad, 4.0 * mad)
    return float(r.square().mean().sqrt())


def coarse_axis_scores(rgb: torch.Tensor, analysis_size: int = 384) -> tuple[float, float]:
    """Return (horizontal-band score, vertical-band score).

    The statistic is intentionally only a tie-breaker for nearly square images.
    For ordinary portrait/landscape photographs the physical sensor-orientation
    prior is more reliable than whole-scene profile statistics, which can be
    dominated by furniture, doors, horizons, etc.
    """
    if rgb.ndim == 3:
        x = rgb.unsqueeze(0)
    elif rgb.ndim == 4 and rgb.shape[0] == 1:
        x = rgb
    else:
        raise ValueError("coarse_axis_scores expects CHW or 1xCHW RGB")
    x = x.detach().float().cpu()
    h, w = x.shape[-2:]
    scale = min(1.0, float(max(64, analysis_size)) / float(max(h, w)))
    if scale < 1.0:
        nh, nw = max(32, int(round(h * scale))), max(32, int(round(w * scale)))
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    y, c = rgb_to_y_cbcr(x)
    # Avoid frame borders, which frequently contain unrelated high-contrast
    # structure and are poor evidence for flicker orientation.
    hh, ww = y.shape[-2:]
    y0, y1 = int(round(hh * 0.05)), int(round(hh * 0.95))
    x0, x1 = int(round(ww * 0.05)), int(round(ww * 0.95))
    y = y[..., y0:y1, x0:x1]
    c = c[..., y0:y1, x0:x1]

    # Horizontal bands vary over rows, therefore median over X.  Vertical bands
    # vary over columns, therefore median over Y. Chroma receives a modest weight
    # because real scene composition is usually stronger in luminance.
    row_y = y.median(dim=-1).values[0, 0]
    row_cb = c[:, 0:1].median(dim=-1).values[0, 0]
    row_cr = c[:, 1:2].median(dim=-1).values[0, 0]
    col_y = y.median(dim=-2).values[0, 0]
    col_cb = c[:, 0:1].median(dim=-2).values[0, 0]
    col_cr = c[:, 1:2].median(dim=-2).values[0, 0]
    hs = _poly_detrended_rms(row_y) + 0.45 * (_poly_detrended_rms(row_cb) + _poly_detrended_rms(row_cr))
    vs = _poly_detrended_rms(col_y) + 0.45 * (_poly_detrended_rms(col_cb) + _poly_detrended_rms(col_cr))
    return float(hs), float(vs)


def decide_band_axis(
    rgb: torch.Tensor,
    requested: str = "auto",
    *,
    portrait_ratio: float = 1.10,
    analysis_size: int = 384,
) -> BandAxisDecision:
    """Choose visible band orientation.

    In auto mode, portrait/landscape geometry is the primary physical prior:
    rotating the camera rotates sensor rows in the displayed image.  A content
    statistic is calculated for diagnostics and is used only for near-square
    images, where aspect ratio carries little orientation information.
    """
    if requested not in {"auto", "horizontal", "vertical"}:
        raise ValueError("requested axis must be auto, horizontal, or vertical")
    if portrait_ratio <= 1.0:
        raise ValueError("portrait_ratio must be > 1")
    if rgb.ndim == 4:
        h, w = rgb.shape[-2:]
    elif rgb.ndim == 3:
        h, w = rgb.shape[-2:]
    else:
        raise ValueError("RGB tensor must be CHW or BCHW")
    aspect = float(h) / float(max(1, w))
    hs, vs = coarse_axis_scores(rgb, analysis_size=analysis_size)

    if requested != "auto":
        return BandAxisDecision(requested, "manual override", hs, vs, aspect)

    if aspect >= float(portrait_ratio):
        return BandAxisDecision(
            "vertical",
            f"portrait geometry H/W={aspect:.3f} >= {portrait_ratio:.3f}; normalize rotated sensor rows",
            hs, vs, aspect,
        )
    if aspect <= 1.0 / float(portrait_ratio):
        return BandAxisDecision(
            "horizontal",
            f"landscape geometry H/W={aspect:.3f} <= {1.0/portrait_ratio:.3f}; native row orientation",
            hs, vs, aspect,
        )

    # Near-square images: there is no useful orientation prior, so use the
    # coarse striping statistic. Preserve legacy horizontal behavior on ties.
    axis = "vertical" if vs > hs * 1.05 else "horizontal"
    return BandAxisDecision(
        axis,
        f"near-square geometry; content scores horizontal={hs:.5f}, vertical={vs:.5f}",
        hs, vs, aspect,
    )


def orient_for_processing(x: torch.Tensor, axis: str) -> torch.Tensor:
    """Rotate visible vertical bands to horizontal bands for row-oriented code."""
    if axis == "horizontal":
        return x
    if axis == "vertical":
        return torch.rot90(x, 1, dims=(-2, -1))
    raise ValueError("axis must be horizontal or vertical")


def restore_display_orientation(x: torch.Tensor, axis: str) -> torch.Tensor:
    if axis == "horizontal":
        return x
    if axis == "vertical":
        return torch.rot90(x, 3, dims=(-2, -1))
    raise ValueError("axis must be horizontal or vertical")
