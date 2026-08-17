#!/usr/bin/env python3
"""Edge-aware residual-band cleanup for low-detail / near-uniform regions.

The local flat-region stage estimates only a low-frequency, row-coherent
Y/CbCr correction.  Its vertical estimator is edge-conductive: masked pixels
do not contribute, and evidence is attenuated when crossing strong scene
boundaries so real horizontal wall/furniture structure is not mistaken for a
flicker band.  A broad structure-density safety gate also suppresses local
correction around long high-contrast fixtures/frames whose support geometry can
otherwise imitate a band.  The estimated/applied correction fields may be
regularized across processing X.  The image itself is never blurred.

The optional residual-profile stage keeps one robust global band period, phase
and waveform, but can fit a slowly varying local amplitude for that waveform.
This lets foreground/background surfaces receive different correction strength
without fitting an independent profile to each object or tile.  The dominant
large-surface equalizer remains a separate opt-in stage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from color_space import y_cbcr_to_rgb


@dataclass
class FlatFilterStats:
    flat_fraction: float
    support_fraction: float
    coarse_fraction: float
    edge_support_fraction: float
    luma_gate_fraction: float
    luma_min_lstar: float | None
    luma_max_lstar: float | None
    band_period_px: float
    sigma_y_px: float
    y_delta_rms: float
    c_delta_rms: float
    profile_y_rms: float
    profile_c_rms: float
    profile_period_px: float
    band_confidence: float
    profile_support_fraction: float
    profile_apply_fraction: float
    profile_confidence_mean: float
    local_extent_fraction: float
    local_color_fraction: float
    local_fill_fraction: float
    local_edge_distance_fraction: float
    local_surface_fraction: float
    surface_equalizer_fraction: float
    surface_equalizer_y_rms: float
    surface_equalizer_c_rms: float
    band_candidates: tuple[tuple[float, float], ...]


def _smoothstep(x: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
    if edge1 <= edge0:
        return (x >= edge1).to(x.dtype)
    t = ((x - float(edge0)) / float(edge1 - edge0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def rgb_to_lab_lstar(rgb: torch.Tensor) -> torch.Tensor:
    """Return CIE Lab L* (D65) as Bx1xHxW from sRGB in [0,1]."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB Bx3xHxW, got {tuple(rgb.shape)}")
    lin = _srgb_to_linear(rgb.float())
    yy = 0.2126729 * lin[:, 0:1] + 0.7151522 * lin[:, 1:2] + 0.0721750 * lin[:, 2:3]
    delta = 6.0 / 29.0
    eps = delta ** 3
    kappa = 1.0 / (3.0 * delta * delta)
    f = torch.where(yy > eps, yy.clamp_min(1e-12).pow(1.0 / 3.0), kappa * yy + 4.0 / 29.0)
    return 116.0 * f - 16.0


def hex_to_lab_lstar(value: str | None) -> float | None:
    """Convert an sRGB hex color to its CIE Lab L* threshold."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"off", "none", "disable", "disabled"}:
        return None
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"Expected #RRGGBB hex color or 'off', got {value!r}")
    vals = [int(text[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    rgb = torch.tensor(vals, dtype=torch.float32).view(1, 3, 1, 1)
    return float(rgb_to_lab_lstar(rgb).item())


def _kernel1d(sigma: float, radius: int, *, device, dtype) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum()


def _smooth_axis(x: torch.Tensor, sigma: float, axis: str) -> torch.Tensor:
    if sigma <= 0:
        return x
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got {tuple(x.shape)}")
    n = x.shape[-1] if axis == "x" else x.shape[-2]
    if n < 2:
        return x
    radius = min(max(1, int(math.ceil(3.0 * sigma))), n - 1)
    k = _kernel1d(sigma, radius, device=x.device, dtype=x.dtype)
    c = x.shape[1]
    if axis == "x":
        p = F.pad(x, (radius, radius, 0, 0), mode="reflect")
        return F.conv2d(p, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    if axis == "y":
        p = F.pad(x, (0, 0, radius, radius), mode="reflect")
        return F.conv2d(p, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    raise ValueError("axis must be 'x' or 'y'")


def _normalized_smooth_axis(
    x: torch.Tensor,
    support: torch.Tensor,
    sigma: float,
    axis: str,
    *,
    min_den: float = 1e-4,
) -> torch.Tensor:
    """Masked normalized Gaussian convolution.

    Unsupported pixels do not contribute to neighboring estimates. This is the
    key anti-halo operation used around black objects, doors, highlights, etc.
    """
    if sigma <= 0:
        return x
    if support.ndim != 4 or support.shape[1] != 1 or support.shape[0] != x.shape[0] or support.shape[-2:] != x.shape[-2:]:
        raise ValueError("support must be Bx1xHxW and match x")
    w = support.float().clamp(0.0, 1.0)
    wx = w.expand(-1, x.shape[1], -1, -1)
    num = _smooth_axis(x.float() * wx, sigma, axis)
    den = _smooth_axis(w, sigma, axis)
    est = num / den.expand_as(num).clamp_min(min_den)
    return torch.where(den.expand_as(num) >= min_den, est, x.float())




def _edge_aware_normalized_smooth_y(
    x: torch.Tensor,
    support: torch.Tensor,
    edge_support: torch.Tensor,
    sigma: float,
    *,
    edge_power: float = 2.0,
    min_den: float = 1e-4,
) -> torch.Tensor:
    """Symmetric edge-aware smoothing along Y with masked observations.

    ``_normalized_smooth_axis(..., "y")`` excludes scene-edge pixels from
    the estimate, but a wide Gaussian can still bridge across the excluded
    rows.  That is dangerous when a real horizontal wall/furniture boundary
    happens to resemble a flicker band.  This recursive smoother treats the
    existing soft ``edge_support`` map as *conductivity*: evidence decays
    normally inside a surface, but propagation is attenuated when crossing a
    strong scene boundary.  Numerator and denominator are filtered separately,
    so unsupported pixels never become observations.

    The source image is not blurred; this is used only to estimate the local
    correction field.
    """
    if sigma <= 0:
        return x.float()
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got {tuple(x.shape)}")
    if support.ndim != 4 or support.shape[1] != 1 or support.shape[0] != x.shape[0] or support.shape[-2:] != x.shape[-2:]:
        raise ValueError("support must be Bx1xHxW and match x")
    if edge_support.ndim != 4 or edge_support.shape[1] != 1 or edge_support.shape[0] != x.shape[0] or edge_support.shape[-2:] != x.shape[-2:]:
        raise ValueError("edge_support must be Bx1xHxW and match x")
    h = x.shape[-2]
    if h < 2:
        return x.float()

    # For symmetric exponential weights a^|d|,
    # variance = 2a/(1-a)^2.  Solve for a so ``sigma`` keeps approximately
    # the same spatial meaning as the Gaussian used by the previous filter.
    s2 = float(sigma) * float(sigma)
    alpha = (s2 + 1.0 - math.sqrt(2.0 * s2 + 1.0)) / s2

    w = support.float().clamp(0.0, 1.0)
    obs_num = x.float() * w.expand(-1, x.shape[1], -1, -1)
    obs_den = w
    conductivity = torch.minimum(edge_support[:, :, 1:, :], edge_support[:, :, :-1, :]).float().clamp(0.0, 1.0)
    if edge_power != 1.0:
        conductivity = conductivity.pow(float(edge_power))

    f_num = obs_num.clone()
    f_den = obs_den.clone()
    for yi in range(1, h):
        a = (float(alpha) * conductivity[:, :, yi-1:yi, :])
        f_num[:, :, yi:yi+1, :] = obs_num[:, :, yi:yi+1, :] + a.expand(-1, x.shape[1], -1, -1) * f_num[:, :, yi-1:yi, :]
        f_den[:, :, yi:yi+1, :] = obs_den[:, :, yi:yi+1, :] + a * f_den[:, :, yi-1:yi, :]

    b_num = obs_num.clone()
    b_den = obs_den.clone()
    for yi in range(h - 2, -1, -1):
        a = (float(alpha) * conductivity[:, :, yi:yi+1, :])
        b_num[:, :, yi:yi+1, :] = obs_num[:, :, yi:yi+1, :] + a.expand(-1, x.shape[1], -1, -1) * b_num[:, :, yi+1:yi+2, :]
        b_den[:, :, yi:yi+1, :] = obs_den[:, :, yi:yi+1, :] + a * b_den[:, :, yi+1:yi+2, :]

    num = f_num + b_num - obs_num
    den = f_den + b_den - obs_den
    est = num / den.expand_as(num).clamp_min(float(min_den))
    return torch.where(den.expand_as(num) >= float(min_den), est, x.float())

def _adaptive_profile_amplitude(
    residual: torch.Tensor,
    correction: torch.Tensor,
    support: torch.Tensor,
    edge_support: torch.Tensor,
    *,
    period: float,
    sigma_x_ratio: float = 0.35,
    sigma_y_ratio: float = 1.50,
    corr_low: float = 0.15,
    corr_high: float = 0.45,
    max_gain: float = 1.00,
    narrow_ratio: float = 0.035,
    base_ratio: float = 0.40,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a slowly varying local gain for a globally estimated band waveform.

    The global residual profile still supplies period, phase and waveform.  This
    stage fits only its amplitude from local covariance, reducing correction in
    regions that do not corroborate the waveform while allowing stronger
    correction on surfaces where the same flicker is clearly present.
    """
    if correction.shape[-1] == 1 and residual.shape[-1] != 1:
        correction = correction.expand(-1, correction.shape[1], -1, residual.shape[-1])
    if correction.shape != residual.shape:
        raise ValueError("adaptive profile correction must match residual")
    if support.ndim != 4 or support.shape[1] != 1 or support.shape[-2:] != residual.shape[-2:]:
        raise ValueError("adaptive profile support must be Bx1xHxW")

    # Keep the fit spatially smooth rather than segmenting it into hard edge
    # domains: a hard amplitude jump is itself visible as a vertical seam.
    # Strong scene edges already reduce ``support``; weighting them a second
    # time makes cross-object evidence weak while retaining a feathered field.
    w = (support.float().clamp(0.0, 1.0) * edge_support.float().clamp(0.0, 1.0)).clamp(0.0, 1.0)
    wc = w.expand(-1, residual.shape[1], -1, -1)
    corr = correction.float()

    # Fit against the same band-limited component used to construct the global
    # profile, not against the full local residual.  The previous v0.38 fit used
    # ``-residual`` directly, so broad wall shading, faces, clothing and other
    # low-frequency scene structure could accidentally correlate with the band
    # waveform and appear as patch-shaped gain islands.
    narrow_sigma = max(0.75, float(period) * float(narrow_ratio))
    base_sigma = max(narrow_sigma + 0.5, float(period) * float(base_ratio))
    target = _smooth_axis(residual.float(), base_sigma, "y") - _smooth_axis(
        residual.float(), narrow_sigma, "y"
    )

    # Sum channels before fitting for CbCr so one scalar amplitude preserves the
    # chroma profile direction instead of letting Cb and Cr diverge independently.
    num = (target * corr * wc).sum(dim=1, keepdim=True)
    den = (corr.square() * wc).sum(dim=1, keepdim=True)
    power = (target.square() * wc).sum(dim=1, keepdim=True)
    coverage = w

    # The amplitude envelope must vary much more slowly than the flicker itself.
    # In particular, give the vertical fit at least roughly a multi-cycle support
    # window so it cannot chase individual bright/dark bands and turn them into
    # local blotches.
    sx = max(16.0, float(period) * float(sigma_x_ratio))
    sy = max(24.0, float(period) * float(sigma_y_ratio))
    for axis, sigma in (("x", sx), ("y", sy)):
        num = _smooth_axis(num, sigma, axis)
        den = _smooth_axis(den, sigma, axis)
        power = _smooth_axis(power, sigma, axis)
        coverage = _smooth_axis(coverage, sigma, axis)

    fit = num / den.clamp_min(1e-8)
    corrcoef = num.abs() / torch.sqrt((den * power).clamp_min(1e-12))
    evidence = _smoothstep(corrcoef, float(corr_low), float(corr_high))
    evidence = evidence * _smoothstep(coverage, 0.08, 0.35)

    # Adaptive mode is an attenuation map by default.  It may reduce a global
    # profile where the local scene does not support it, but it no longer silently
    # boosts a user-requested 1.0 profile to 1.5 in isolated regions.  Users who
    # explicitly want amplification can still raise --flat-profile-adaptive-max-gain.
    amp = fit.clamp(0.0, float(max_gain)) * evidence
    return amp.clamp(0.0, float(max_gain)), corrcoef.clamp(0.0, 1.0)


def _profile_no_harm_gate(
    residual: torch.Tensor,
    applied_delta: torch.Tensor,
    support: torch.Tensor,
    edge_support: torch.Tensor,
    *,
    period: float,
    narrow_ratio: float,
    base_ratio: float,
) -> torch.Tensor:
    """Return a band-coherent no-harm gate for residual-profile cleanup.

    A residual profile is physically a one-dimensional waveform repeated along
    the band direction. The previous safety gate was fully 2-D; although it
    could reject a locally harmful correction, its 2-D mask could trace blurred
    people/objects and make those silhouettes visible in the correction field.

    This validator therefore compares before/after band energy locally *along
    the band direction only*. Evidence is collapsed across the waveform axis
    and the resulting gain is broadly smoothed along X, then expanded over Y.
    It can still attenuate the profile on different parts of a scene (important
    after vertical-band axis normalization), but it cannot draw object-shaped
    contours or modulate individual bright/dark rows.
    """
    if residual.shape != applied_delta.shape:
        raise ValueError("profile no-harm residual and applied_delta must match")
    if support.ndim != 4 or support.shape[1] != 1 or support.shape[-2:] != residual.shape[-2:]:
        raise ValueError("profile no-harm support must be Bx1xHxW")

    narrow_sigma = max(0.75, float(period) * float(narrow_ratio))
    base_sigma = max(narrow_sigma + 0.5, float(period) * float(base_ratio))

    def band_component(x: torch.Tensor) -> torch.Tensor:
        return _smooth_axis(x.float(), narrow_sigma, "y") - _smooth_axis(x.float(), base_sigma, "y")

    before = band_component(residual)
    after = band_component(residual.float() + applied_delta.float())
    before_e = before.square().mean(dim=1, keepdim=True)
    after_e = after.square().mean(dim=1, keepdim=True)

    # Collapse validation across the waveform axis (Y). The gate may vary only
    # along X, i.e. along the displayed band direction after axis normalization.
    # This is the critical anti-silhouette constraint.
    w = (support.float().clamp(0.0, 1.0) * edge_support.float().clamp(0.0, 1.0)).clamp(0.0, 1.0)
    den = w.sum(dim=-2, keepdim=True)
    before_x = (before_e * w).sum(dim=-2, keepdim=True) / den.clamp_min(1e-6)
    after_x = (after_e * w).sum(dim=-2, keepdim=True) / den.clamp_min(1e-6)
    coverage_x = w.mean(dim=-2, keepdim=True)

    # Keep the gain much broader than ordinary object edges. It is allowed to
    # distinguish large scene zones, not individual people, panels, or bands.
    sx = max(24.0, min(128.0, float(period) * 0.35))
    before_x = _smooth_axis(before_x, sx, "x")
    after_x = _smooth_axis(after_x, sx, "x")
    coverage_x = _smooth_axis(coverage_x, sx, "x")

    improvement = (before_x - after_x) / before_x.clamp_min(1e-10)
    gate_x = _smoothstep(improvement, 0.01, 0.08)
    gate_x = gate_x * _smoothstep(coverage_x, 0.03, 0.15)
    gate_x = torch.where(before_x > 1e-10, gate_x, torch.zeros_like(gate_x))
    return gate_x.clamp(0.0, 1.0).expand(-1, 1, residual.shape[-2], -1)


def _pad_dx(x: torch.Tensor) -> torch.Tensor:
    d = x[..., 1:] - x[..., :-1]
    return F.pad(d, (0, 1, 0, 0), mode="replicate")


def _pad_dy(x: torch.Tensor) -> torch.Tensor:
    d = x[..., 1:, :] - x[..., :-1, :]
    return F.pad(d, (0, 0, 0, 1), mode="replicate")


def make_scene_edge_support(
    y: torch.Tensor,
    c: torch.Tensor,
    *,
    preblur_sigma: float = 1.5,
    edge_low: float = 0.018,
    edge_high: float = 0.055,
    guard_px: int = 2,
) -> torch.Tensor:
    """Soft support barrier around strong scene boundaries.

    It is used for *estimation*, not final blending. Therefore an excluded
    object cannot bleed into a neighboring wall through the Gaussian kernels,
    while the separately feathered blend mask can still approach the edge.
    """
    yy = y.float()
    cc = c.float()
    if preblur_sigma > 0:
        yy = _smooth_axis(_smooth_axis(yy, preblur_sigma, "x"), preblur_sigma, "y")
        cc = _smooth_axis(_smooth_axis(cc, preblur_sigma, "x"), preblur_sigma, "y")
    gx = torch.sqrt(_pad_dx(yy).square() + 0.5 * _pad_dx(cc).square().mean(dim=1, keepdim=True) + 1e-12)
    gy = torch.sqrt(_pad_dy(yy).square() + 0.5 * _pad_dy(cc).square().mean(dim=1, keepdim=True) + 1e-12)
    g = torch.maximum(gx, gy)
    edge = _smoothstep(g, edge_low, edge_high)
    if guard_px > 0:
        k = 2 * int(guard_px) + 1
        edge = F.max_pool2d(edge, kernel_size=k, stride=1, padding=int(guard_px))
    return (1.0 - edge).clamp(0.0, 1.0)


def make_broad_structure_support(
    y: torch.Tensor,
    c: torch.Tensor,
    *,
    sigma_px: float = 8.0,
    edge_low: float = 0.025,
    edge_high: float = 0.080,
    guard_px: int = 8,
    feather_sigma: float = 6.0,
) -> torch.Tensor:
    """Protect soft/defocused scene boundaries from the local flat filter.

    The normal edge barrier is intentionally sensitive to fairly local
    gradients. A heavily defocused person, wall moulding, or panel transition
    can spread the same contrast over tens of pixels and fall below that
    per-pixel threshold, making real structure look like a flat surface.

    Here the image is inspected at a broader scale and the gradient is
    scale-normalized by ``sigma_px``. This makes a blurred step remain visible
    while faint residual rolling bands, whose post-neural per-pixel gradients
    are much smaller, usually remain below the guard. The result is used only
    by the local flat-region stage; the texture-tolerant residual profile keeps
    its existing support model.
    """
    if sigma_px <= 0:
        return torch.ones_like(y)
    yy = _smooth_axis(_smooth_axis(y.float(), sigma_px, "x"), sigma_px, "y")
    cc = _smooth_axis(_smooth_axis(c.float(), sigma_px, "x"), sigma_px, "y")
    gx = torch.sqrt(_pad_dx(yy).square() + 0.5 * _pad_dx(cc).square().mean(dim=1, keepdim=True) + 1e-12)
    gy = torch.sqrt(_pad_dy(yy).square() + 0.5 * _pad_dy(cc).square().mean(dim=1, keepdim=True) + 1e-12)
    g = torch.maximum(gx, gy) * float(sigma_px)
    edge = _smoothstep(g, float(edge_low), float(edge_high))
    if guard_px > 0:
        r = int(guard_px)
        edge = F.max_pool2d(edge, kernel_size=2 * r + 1, stride=1, padding=r)
    support = (1.0 - edge).clamp(0.0, 1.0)
    if feather_sigma > 0:
        support = _smooth_axis(_smooth_axis(support, float(feather_sigma), "x"), float(feather_sigma), "y")
    return support.clamp(0.0, 1.0)


def _flat_metric(
    y: torch.Tensor,
    c: torch.Tensor,
    *,
    coherence_sigma_x: float,
    metric_sigma: float,
    preblur_sigma: float = 0.0,
) -> torch.Tensor:
    yy = y.float()
    cc = c.float()
    if preblur_sigma > 0:
        yy = _smooth_axis(_smooth_axis(yy, preblur_sigma, "x"), preblur_sigma, "y")
        cc = _smooth_axis(_smooth_axis(cc, preblur_sigma, "x"), preblur_sigma, "y")

    dx_y = _pad_dx(yy).abs()
    dx_c = _pad_dx(cc).abs().mean(dim=1, keepdim=True)
    dy_y = _pad_dy(yy)
    dy_c = _pad_dy(cc)
    # X-coherent vertical variation is allowed because it is characteristic of
    # rolling bands rather than local scene texture.
    scene_dy_y = (dy_y - _smooth_axis(dy_y, coherence_sigma_x, "x")).abs()
    scene_dy_c = (dy_c - _smooth_axis(dy_c, coherence_sigma_x, "x")).abs().mean(dim=1, keepdim=True)
    metric = dx_y + 0.55 * dx_c + 0.85 * scene_dy_y + 0.45 * scene_dy_c
    if metric_sigma > 0:
        metric = _smooth_axis(_smooth_axis(metric, metric_sigma, "x"), metric_sigma, "y")
    return metric


def make_flat_mask(
    y: torch.Tensor,
    c: torch.Tensor,
    *,
    coherence_sigma_x: float = 16.0,
    metric_sigma: float = 1.5,
    flat_full: float = 0.007,
    flat_none: float = 0.025,
    preblur_sigma: float = 0.0,
) -> torch.Tensor:
    """Return a soft Bx1xHxW flatness mask."""
    if y.ndim != 4 or y.shape[1] != 1 or c.ndim != 4 or c.shape[1] != 2:
        raise ValueError("Expected Y Bx1xHxW and C Bx2xHxW")
    if flat_none <= flat_full:
        raise ValueError("flat_none must be greater than flat_full")
    metric = _flat_metric(
        y, c,
        coherence_sigma_x=coherence_sigma_x,
        metric_sigma=metric_sigma,
        preblur_sigma=preblur_sigma,
    )
    mask = ((flat_none - metric) / (flat_none - flat_full)).clamp(0.0, 1.0)
    return mask * mask * (3.0 - 2.0 * mask)


def _gate_from_lstar(
    lstar: torch.Tensor,
    *,
    highpass: str | None,
    lowpass: str | None,
    shadow_ramp_lstar: float,
    highlight_ramp_lstar: float,
) -> tuple[torch.Tensor, float | None, float | None]:
    if shadow_ramp_lstar < 0 or highlight_ramp_lstar < 0:
        raise ValueError("Lab-lightness ramps must be >= 0")
    lo = hex_to_lab_lstar(highpass)
    hi = hex_to_lab_lstar(lowpass)
    if lo is not None and hi is not None and hi <= lo:
        raise ValueError(f"lowpass L* ({hi:.3f}) must be greater than highpass L* ({lo:.3f})")
    gate = torch.ones_like(lstar)
    if lo is not None:
        half = 0.5 * float(shadow_ramp_lstar)
        if shadow_ramp_lstar == 0:
            gate = gate * (lstar >= lo).to(gate.dtype)
        else:
            gate = gate * _smoothstep(lstar, lo - half, lo + half)
    if hi is not None:
        half = 0.5 * float(highlight_ramp_lstar)
        if highlight_ramp_lstar == 0:
            gate = gate * (lstar <= hi).to(gate.dtype)
        else:
            gate = gate * (1.0 - _smoothstep(lstar, hi - half, hi + half))
    return gate.clamp(0.0, 1.0), lo, hi


def make_perceptual_luma_gate(
    y: torch.Tensor,
    c: torch.Tensor,
    *,
    highpass: str | None = "#232323",
    lowpass: str | None = "#efefef",
    shadow_ramp_lstar: float = 12.0,
    highlight_ramp_lstar: float = 8.0,
    spatial_feather: float = 1.25,
    base_lstar: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float | None, float | None]:
    """Soft tone eligibility gate using raw or supplied band-resistant Lab L*."""
    if base_lstar is None:
        rgb = y_cbcr_to_rgb(y.float(), c.float()).clamp(0.0, 1.0)
        base_lstar = rgb_to_lab_lstar(rgb)
    gate, lo, hi = _gate_from_lstar(
        base_lstar,
        highpass=highpass,
        lowpass=lowpass,
        shadow_ramp_lstar=shadow_ramp_lstar,
        highlight_ramp_lstar=highlight_ramp_lstar,
    )
    if spatial_feather > 0:
        gate = _smooth_axis(_smooth_axis(gate, spatial_feather, "x"), spatial_feather, "y")
    return gate.clamp(0.0, 1.0), lo, hi


def make_band_resistant_base_lstar(
    y: torch.Tensor,
    c: torch.Tensor,
    *,
    band_period_px: float,
    edge_support: torch.Tensor,
    highpass: str | None,
    lowpass: str | None,
    shadow_ramp_lstar: float,
    highlight_ramp_lstar: float,
    sigma_ratio: float = 0.40,
    sigma_x: float = 2.0,
) -> torch.Tensor:
    """Estimate surface/base L* while attenuating the rolling-band cycle.

    A relaxed raw-L* eligibility support prevents deep black / clipped objects
    from contaminating the estimate. This avoids the circular failure where a
    flicker trough is rejected merely because the trough itself made the wall
    darker than the threshold.
    """
    rgb = y_cbcr_to_rgb(y.float(), c.float()).clamp(0.0, 1.0)
    lstar = rgb_to_lab_lstar(rgb)
    raw_gate, _, _ = _gate_from_lstar(
        lstar,
        highpass=highpass,
        lowpass=lowpass,
        # Broader preliminary ramps make this support tolerant of flicker troughs.
        shadow_ramp_lstar=max(18.0, shadow_ramp_lstar * 1.5),
        highlight_ramp_lstar=max(12.0, highlight_ramp_lstar * 1.5),
    )
    base_support = (edge_support * raw_gate).clamp(0.0, 1.0)
    base = _normalized_smooth_axis(lstar, base_support, sigma_x, "x") if sigma_x > 0 else lstar
    sigma_y = max(1.0, float(band_period_px) * float(sigma_ratio))
    sigma_y = min(sigma_y, max(1.0, y.shape[-2] / 8.0))
    return _normalized_smooth_axis(base, base_support, sigma_y, "y")


def _resize_row_signal(row: torch.Tensor, target_h: int) -> torch.Tensor:
    """Resize a 1-D row signal to target_h while preserving batch/channel axes."""
    if row.ndim == 1:
        row = row.view(1, 1, -1, 1)
    elif row.ndim == 2:
        row = row.unsqueeze(1).unsqueeze(-1)
    elif row.ndim == 3:
        row = row.unsqueeze(-1)
    elif row.ndim != 4:
        raise ValueError(f"Unsupported row shape {tuple(row.shape)}")
    if row.shape[-2] == target_h:
        return row
    return F.interpolate(row, size=(target_h, 1), mode="bilinear", align_corners=False)


def _remove_linear_y_trend(x: torch.Tensor, support: torch.Tensor | None = None) -> torch.Tensor:
    """Remove a weighted linear vertical trend independently in each column."""
    if x.ndim != 4:
        raise ValueError("Expected BCHW tensor")
    b, c, h, w = x.shape
    t = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
    if support is None:
        sw = torch.ones((b, 1, h, w), device=x.device, dtype=x.dtype)
    else:
        sw = support.float().clamp(0.0, 1.0)
    ww = sw.expand(-1, c, -1, -1)
    S0 = ww.sum(dim=-2, keepdim=True).clamp_min(1e-6)
    S1 = (ww * t).sum(dim=-2, keepdim=True)
    S2 = (ww * t * t).sum(dim=-2, keepdim=True)
    X0 = (ww * x).sum(dim=-2, keepdim=True)
    X1 = (ww * x * t).sum(dim=-2, keepdim=True)
    den = (S0 * S2 - S1 * S1).clamp_min(1e-6)
    slope = (S0 * X1 - S1 * X0) / den
    intercept = (X0 - slope * S1) / S0
    return x - (intercept + slope * t)


def _robust_row_consensus(
    x: torch.Tensor,
    support: torch.Tensor,
    *,
    huber_k: float = 2.5,
    min_coverage: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Robust row consensus tolerant of texture, noise, and minority objects.

    Returns (row, valid, confidence, robust_spread).  The first estimate is a
    masked median; a Cauchy/Huber-like reweighting around that median then
    recovers a smoother mean without letting outliers dominate.
    """
    if x.ndim != 4 or support.ndim != 4 or support.shape[1] != 1:
        raise ValueError("Expected x BCHW and support Bx1xHxW")
    w = support.float().clamp(0.0, 1.0).expand(-1, x.shape[1], -1, -1)
    nan = torch.full_like(x, float("nan"))
    masked = torch.where(w > 0.05, x.float(), nan)
    med = torch.nanmedian(masked, dim=-1, keepdim=True).values
    med = torch.nan_to_num(med, nan=0.0)
    absdev = (x.float() - med).abs()
    mad_masked = torch.where(w > 0.05, absdev, nan)
    mad = torch.nanmedian(mad_masked, dim=-1, keepdim=True).values
    mad = torch.nan_to_num(mad, nan=0.0).clamp_min(1e-5)
    scale = (1.4826 * mad).clamp_min(1e-5)
    z = (x.float() - med) / (float(huber_k) * scale + 1e-6)
    robust_w = w / (1.0 + z.square())
    den = robust_w.sum(dim=-1, keepdim=True)
    row = (x.float() * robust_w).sum(dim=-1, keepdim=True) / den.clamp_min(1e-6)

    coverage = support.float().mean(dim=-1, keepdim=True)
    valid = (coverage >= float(min_coverage)).to(x.dtype)
    # Agreement is relative to the typical robust spread of this image, so a
    # naturally noisy/textured wall can still achieve high confidence.
    spread = mad.mean(dim=1, keepdim=True)
    typical = torch.nanmedian(spread.reshape(spread.shape[0], -1), dim=-1).values
    typical = torch.nan_to_num(typical, nan=0.01).view(-1, 1, 1, 1).clamp_min(1e-4)
    agreement = 1.0 / (1.0 + spread / (3.0 * typical))
    coverage_conf = _smoothstep(coverage, min_coverage, max(min_coverage + 0.05, 0.40))
    confidence = (coverage_conf * agreement).clamp(0.0, 1.0)
    return row, valid, confidence, spread


def _make_image_row_signals(
    y: torch.Tensor,
    c: torch.Tensor,
    support: torch.Tensor,
    *,
    max_h: int = 768,
    max_w: int = 512,
    huber_k: float = 2.5,
) -> list[tuple[torch.Tensor, float]]:
    """Build texture-tolerant residual row signals from the restored image."""
    h, w = y.shape[-2:]
    scale = min(1.0, float(max_h) / h, float(max_w) / w)
    if scale < 1.0:
        hh = max(32, int(round(h * scale)))
        ww = max(32, int(round(w * scale)))
        yy = F.interpolate(y.float(), size=(hh, ww), mode="area")
        cc = F.interpolate(c.float(), size=(hh, ww), mode="area")
        ss = F.interpolate(support.float(), size=(hh, ww), mode="area")
    else:
        yy, cc, ss = y.float(), c.float(), support.float()

    logy = torch.log(yy.clamp_min(0.0) + 0.02)
    logy = _remove_linear_y_trend(logy, ss)
    cc_res = _remove_linear_y_trend(cc, ss)
    row_y, _, conf_y, _ = _robust_row_consensus(logy, ss, huber_k=huber_k)
    row_c, _, conf_c, _ = _robust_row_consensus(cc_res, ss, huber_k=huber_k)
    signals: list[tuple[torch.Tensor, float]] = []
    signals.append((row_y[:, 0:1], 0.80 * float(conf_y.mean())))
    signals.append((row_c[:, 0:1], 0.55 * float(conf_c.mean())))
    signals.append((row_c[:, 1:2], 0.55 * float(conf_c.mean())))
    return signals


def _spectral_power(row: torch.Tensor, analysis_h: int) -> torch.Tensor | None:
    row = _resize_row_signal(row.float(), analysis_h).view(-1)
    if row.numel() < 16:
        return None
    # Only remove offset + linear composition drift.  Do not aggressively
    # high-pass: broad rolling bands are precisely what v1.6 must retain.
    t = torch.linspace(-1.0, 1.0, row.numel(), device=row.device, dtype=row.dtype)
    A = torch.stack((torch.ones_like(t), t), dim=1)
    sol = torch.linalg.lstsq(A, row.unsqueeze(1)).solution.squeeze(1)
    row = row - (A @ sol)
    std = row.std().clamp_min(1e-7)
    row = row / std
    # A mild Tukey-like window reduces endpoint leakage without killing broad
    # components as strongly as a full Hann window.
    n = row.numel()
    edge = max(1, n // 10)
    win = torch.ones_like(row)
    if edge > 1:
        ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0, math.pi, edge, device=row.device, dtype=row.dtype))
        win[:edge] = ramp
        win[-edge:] = torch.flip(ramp, dims=[0])
    spec = torch.fft.rfft(row * win)
    power = spec.real.square() + spec.imag.square()
    if power.numel() < 3 or float(power[1:].max()) <= 1e-12:
        return None
    power[0] = 0
    return power / power[1:].max().clamp_min(1e-12)


@dataclass
class BandPeriodEstimate:
    period_px: float
    confidence: float
    candidates: tuple[tuple[float, float], ...]


def estimate_band_period_multiscale(
    luma_field: torch.Tensor | None,
    chroma_delta: torch.Tensor | None,
    *,
    full_height: int,
    y: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    support: torch.Tensor | None = None,
    min_period_px: float = 12.0,
    max_period_fraction: float = 0.60,
    analysis_h: int = 512,
    huber_k: float = 2.5,
) -> BandPeriodEstimate:
    """Multiscale, harmonic-aware vertical band-period estimator.

    Model correction maps are the primary signal.  When restored Y/CbCr are
    supplied, a robust row-consensus residual adds evidence without requiring a
    texture-free/flat surface.  Candidate fundamentals receive credit for power
    at 2x/3x/4x harmonics, preventing a strong harmonic from automatically
    replacing a broader true band period.
    """
    signals: list[tuple[torch.Tensor, float]] = []
    for x, base_weight in ((luma_field, 1.00), (chroma_delta, 0.80)):
        if x is None or x.ndim != 4:
            continue
        xf = x.detach().float()
        if xf.shape[1] == 1:
            signals.append((xf.mean(dim=-1), base_weight))
        else:
            for ch in range(xf.shape[1]):
                signals.append((xf[:, ch:ch+1].mean(dim=-1), base_weight / math.sqrt(xf.shape[1])))
    if y is not None and c is not None and support is not None:
        signals.extend(_make_image_row_signals(y, c, support, huber_k=huber_k))

    powers = []
    weights = []
    for row, weight in signals:
        if weight <= 1e-4:
            continue
        p = _spectral_power(row, analysis_h)
        if p is not None:
            powers.append(p)
            weights.append(float(weight))
    if not powers:
        fallback = max(min_period_px, min(float(full_height) / 8.0, 128.0))
        return BandPeriodEstimate(float(fallback), 0.0, ((float(fallback), 0.0),))

    max_len = min(p.numel() for p in powers)
    stack = torch.stack([p[:max_len] for p in powers], dim=0)
    ww = torch.tensor(weights, device=stack.device, dtype=stack.dtype).view(-1, 1)
    combined = (stack * ww).sum(dim=0) / ww.sum().clamp_min(1e-6)
    combined = combined / combined[1:].max().clamp_min(1e-12)

    # DFT bin k corresponds to full-resolution period H/k because every row
    # signal is resampled to the same analysis height.
    min_k = max(1, int(math.ceil(1.0 / max(1e-3, float(max_period_fraction)))))
    max_k = min(combined.numel() - 1, max(min_k, int(math.floor(float(full_height) / max(min_period_px, 1e-3)))))
    scored: list[tuple[float, int]] = []
    for k in range(min_k, max_k + 1):
        base = float(combined[k])
        if base < 0.025:
            continue
        # Fundamental evidence plus a strong but bounded harmonic bonus.
        score = 0.40 * base
        for mult, wt in ((2, 0.40), (3, 0.14), (4, 0.06)):
            kk = k * mult
            if kk < combined.numel():
                score += wt * float(combined[kk])
        # Very small preference for the broader member of an otherwise similar
        # harmonic family; not enough to invent a low-frequency fundamental.
        period = float(full_height) / float(k)
        score *= 1.0 + 0.025 * math.log2(max(period, min_period_px) / max(min_period_px, 1.0))
        scored.append((score, k))

    if not scored:
        k = int(torch.argmax(combined[min_k:max_k+1]).item()) + min_k
        scored = [(float(combined[k]), k)]
    scored.sort(reverse=True)
    best_score, best_k = scored[0]
    best_period = float(full_height) / float(best_k)
    second = scored[1][0] if len(scored) > 1 else 0.0
    peak = float(combined[best_k])
    margin = max(0.0, (best_score - second) / max(best_score, 1e-6))
    confidence = max(0.0, min(1.0, 0.55 * peak + 0.45 * margin))

    # Show distinct top periods; nearby bins are suppressed to keep logs useful.
    chosen: list[tuple[float, float]] = []
    for score, k in scored:
        period = float(full_height) / float(k)
        if any(abs(math.log(period / p)) < 0.10 for p, _ in chosen):
            continue
        chosen.append((period, float(score)))
        if len(chosen) >= 5:
            break
    return BandPeriodEstimate(best_period, confidence, tuple(chosen))


def estimate_band_period(
    luma_field: torch.Tensor | None,
    chroma_delta: torch.Tensor | None,
    *,
    full_height: int,
    min_period_px: float = 6.0,
    max_period_fraction: float = 0.45,
) -> float:
    """v1.5-compatible dominant-period estimate used by the local flattener."""
    signals = []
    work_h = None
    for x in (luma_field, chroma_delta):
        if x is None:
            continue
        xf = x.detach().float()
        if xf.ndim != 4:
            continue
        work_h = xf.shape[-2]
        if xf.shape[1] == 1:
            row = xf.mean(dim=-1).squeeze(1)
        else:
            row = xf.mean(dim=-1)
            row = torch.sqrt((row * row).mean(dim=1) + 1e-12)
        row = row - row.mean(dim=-1, keepdim=True)
        std = row.std(dim=-1, keepdim=True).clamp_min(1e-6)
        signals.append(row / std)
    if not signals or work_h is None or work_h < 16:
        return max(min_period_px, min(float(full_height) / 12.0, 64.0))

    sig = torch.stack(signals, dim=0).mean(dim=0).mean(dim=0)
    trend = _smooth_axis(sig.view(1, 1, -1, 1), max(2.0, work_h / 32.0), "y").view(-1)
    sig = sig - trend
    spec = torch.fft.rfft(sig)
    power = spec.real.square() + spec.imag.square()
    if power.numel() <= 3:
        return max(min_period_px, min(float(full_height) / 12.0, 64.0))

    min_bin = 2
    max_bin = max(min_bin + 1, work_h // 4)
    hi = min(power.numel(), max_bin + 1)
    band = power[min_bin:hi]
    if band.numel() == 0 or float(band.max()) <= 1e-12:
        return max(min_period_px, min(float(full_height) / 12.0, 64.0))
    k = int(torch.argmax(band).item()) + min_bin
    period_work = float(work_h) / float(k)
    scale = float(full_height) / float(work_h)
    period = period_work * scale
    return float(max(min_period_px, min(period, float(full_height) * max_period_fraction)))


def make_large_surface_extent_gate(
    candidate: torch.Tensor,
    *,
    sigma_px: float = 32.0,
    low_fraction: float = 0.18,
    high_fraction: float = 0.50,
) -> torch.Tensor:
    """Reject small isolated locally-flat islands while preserving large surfaces.

    The local flat detector is intentionally semantic-free, so smooth skin or a
    small object patch can occasionally resemble a wall.  This gate asks a
    second question: does the candidate belong to a *large* surrounding surface?
    A broad Gaussian occupancy estimate is converted to a soft gate, therefore
    the safety check does not introduce hard compositing borders.
    """
    if candidate.ndim != 4 or candidate.shape[1] != 1:
        raise ValueError("candidate must be Bx1xHxW")
    if sigma_px <= 0:
        return torch.ones_like(candidate)
    if not (0.0 <= low_fraction < high_fraction <= 1.0):
        raise ValueError("extent fractions must satisfy 0 <= low < high <= 1")
    occ = _smooth_axis(_smooth_axis(candidate.float().clamp(0.0, 1.0), sigma_px, "x"), sigma_px, "y")
    return _smoothstep(occ, low_fraction, high_fraction).clamp(0.0, 1.0)




def make_local_same_surface_gates(
    y: torch.Tensor,
    c: torch.Tensor,
    *,
    base_lstar: torch.Tensor,
    candidate: torch.Tensor,
    edge_support: torch.Tensor,
    color_sigma_px: float = 44.0,
    color_preblur_sigma: float = 2.0,
    luma_tolerance_lstar: float = 7.0,
    chroma_tolerance: float = 0.030,
    fill_sigma_px: float = 36.0,
    fill_low: float = 0.38,
    fill_high: float = 0.68,
    edge_distance_px: int = 6,
    edge_feather_sigma: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return color, fill, edge-distance, and combined local-surface gates.

    A locally smooth patch is not sufficient evidence for wall-style flattening.
    This safety layer additionally requires that the patch agree with the broad
    surrounding surface color, occupy most of a large neighborhood, and sit away
    from a strong object boundary. The gates are soft and affect only the local
    flat filter; the texture-tolerant global profile remains independent.
    """
    if candidate.ndim != 4 or candidate.shape[1] != 1:
        raise ValueError("candidate must be Bx1xHxW")
    if base_lstar.shape != candidate.shape:
        raise ValueError("base_lstar must match candidate")
    if color_sigma_px <= 0:
        color_gate = torch.ones_like(candidate)
    else:
        # Fine texture is irrelevant here; compare a gently preblurred surface
        # color with a much broader candidate-weighted neighborhood color.
        ll = (base_lstar.float() / 100.0).clamp(0.0, 1.0)
        cc = c.float()
        if color_preblur_sigma > 0:
            ll = _smooth_axis(_smooth_axis(ll, color_preblur_sigma, "x"), color_preblur_sigma, "y")
            cc = _smooth_axis(_smooth_axis(cc, color_preblur_sigma, "x"), color_preblur_sigma, "y")
        broad_support = (candidate.float().clamp(0.0, 1.0) * _smoothstep(edge_support, 0.20, 0.85)).clamp(0.0, 1.0)
        mean_l = _normalized_smooth_axis(ll, broad_support, color_sigma_px, "x")
        mean_l = _normalized_smooth_axis(mean_l, broad_support, color_sigma_px, "y")
        mean_c = _normalized_smooth_axis(cc, broad_support, color_sigma_px, "x")
        mean_c = _normalized_smooth_axis(mean_c, broad_support, color_sigma_px, "y")
        dl = (ll - mean_l).abs() * 100.0
        dc = torch.sqrt((cc - mean_c).square().sum(dim=1, keepdim=True) + 1e-12)
        # Tolerances are 50%-ish transition centers with a generous soft shoulder.
        lg = 1.0 - _smoothstep(dl, 0.55 * float(luma_tolerance_lstar), 1.45 * float(luma_tolerance_lstar))
        cg = 1.0 - _smoothstep(dc, 0.55 * float(chroma_tolerance), 1.45 * float(chroma_tolerance))
        color_gate = (lg * cg).clamp(0.0, 1.0)

    same_surface_seed = (candidate.float().clamp(0.0, 1.0) * color_gate).clamp(0.0, 1.0)
    if fill_sigma_px <= 0:
        fill_gate = torch.ones_like(candidate)
    else:
        occ = _smooth_axis(_smooth_axis(same_surface_seed, fill_sigma_px, "x"), fill_sigma_px, "y")
        fill_gate = _smoothstep(occ, float(fill_low), float(fill_high)).clamp(0.0, 1.0)

    if edge_distance_px <= 0:
        edge_distance_gate = torch.ones_like(candidate)
    else:
        # Turn the existing support barrier into a wider local-filter-only guard.
        # The subsequent blur makes the fade gradual instead of drawing a halo.
        edge = (1.0 - edge_support.float().clamp(0.0, 1.0))
        r = int(edge_distance_px)
        k = 2 * r + 1
        expanded = F.max_pool2d(edge, kernel_size=k, stride=1, padding=r)
        edge_distance_gate = (1.0 - expanded).clamp(0.0, 1.0)
        if edge_feather_sigma > 0:
            edge_distance_gate = _smooth_axis(_smooth_axis(edge_distance_gate, edge_feather_sigma, "x"), edge_feather_sigma, "y")

    # color_gate already participates in the fill seed, so multiplying it a
    # second time would over-penalize textured but genuinely uniform walls.
    combined = (fill_gate * edge_distance_gate).clamp(0.0, 1.0)
    return color_gate, fill_gate, edge_distance_gate, combined


def _morph_close_binary(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    k = 2 * int(radius) + 1
    dil = F.max_pool2d(x, kernel_size=k, stride=1, padding=int(radius))
    ero = -F.max_pool2d(-dil, kernel_size=k, stride=1, padding=int(radius))
    return ero


def _largest_component_lowres(mask: torch.Tensor, *, threshold: float = 0.35) -> torch.Tensor:
    """Return the largest 8-connected binary component of a 1x1 low-res mask.

    This tiny CPU flood fill avoids adding scipy/opencv as runtime dependencies.
    It runs only on the equalizer analysis mask (normally <=256 px per side).
    """
    if mask.ndim != 4 or mask.shape[0] != 1 or mask.shape[1] != 1:
        raise ValueError("largest-component helper currently expects 1x1xHxW")
    arr = (mask[0, 0].detach().cpu().numpy() >= float(threshold))
    h, w = arr.shape
    seen = bytearray(h * w)
    best = []
    neigh = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
    from collections import deque
    for yy in range(h):
        base = yy * w
        for xx in range(w):
            idx = base + xx
            if seen[idx] or not arr[yy, xx]:
                continue
            q = deque([(yy, xx)])
            seen[idx] = 1
            comp = []
            while q:
                y0, x0 = q.popleft()
                comp.append((y0, x0))
                for dy, dx in neigh:
                    y1, x1 = y0 + dy, x0 + dx
                    if 0 <= y1 < h and 0 <= x1 < w:
                        j = y1 * w + x1
                        if not seen[j] and arr[y1, x1]:
                            seen[j] = 1
                            q.append((y1, x1))
            if len(comp) > len(best):
                best = comp
    out = torch.zeros_like(mask)
    if best:
        yy = torch.tensor([v[0] for v in best], device=mask.device, dtype=torch.long)
        xx = torch.tensor([v[1] for v in best], device=mask.device, dtype=torch.long)
        out[0, 0, yy, xx] = 1.0
    return out



def make_profile_surface_handoff(
    y: torch.Tensor,
    c: torch.Tensor,
    candidate: torch.Tensor,
    *,
    analysis_size: int = 256,
    component_threshold: float = 0.55,
    close_radius: int = 1,
    min_area_fraction: float = 0.08,
    luma_tolerance: float = 0.30,
    chroma_tolerance: float = 0.06,
    feather_sigma: float = 4.0,
) -> torch.Tensor:
    """Return one dominant smooth-surface ownership map for profile handoff.

    The residual-profile stage may legitimately need a very high strength on a
    foreground subject while the same one-dimensional waveform is excessive on
    a large smooth wall.  A generic blurred flatness mask is unsafe here: its
    blur can leak background ownership into smooth skin and cap the correction
    on the subject.

    Instead, select a single large connected flat-surface component at low
    resolution, estimate that component's dominant Y/CbCr color, reject
    differently colored smooth islands, then re-select the dominant component.
    Only a small final feather is used to hide the analysis-grid boundary.  This
    keeps the handoff on the wall/background instead of bleeding into enclosed
    foreground surfaces such as faces or clothing.
    """
    if candidate.ndim != 4 or candidate.shape[1] != 1:
        raise ValueError("profile handoff candidate must be Bx1xHxW")
    if y.ndim != 4 or y.shape[1] != 1 or y.shape[0] != candidate.shape[0] or y.shape[-2:] != candidate.shape[-2:]:
        raise ValueError("profile handoff Y must match candidate")
    if c.ndim != 4 or c.shape[1] != 2 or c.shape[0] != candidate.shape[0] or c.shape[-2:] != candidate.shape[-2:]:
        raise ValueError("profile handoff CbCr must match candidate")

    h, w = candidate.shape[-2:]
    scale = min(1.0, float(max(32, analysis_size)) / max(h, w))
    ah, aw = max(24, int(round(h * scale))), max(24, int(round(w * scale)))

    cand = candidate.float().clamp(0.0, 1.0)
    # A gentle preblur is used only for dominant-color estimation.  It prevents
    # image grain / fine skin texture from moving the color center.
    ys = _smooth_axis(_smooth_axis(y.float(), 2.0, "x"), 2.0, "y")
    cs = _smooth_axis(_smooth_axis(c.float(), 2.0, "x"), 2.0, "y")

    out_components: list[torch.Tensor] = []
    for bi in range(cand.shape[0]):
        cb = cand[bi:bi+1]
        low = F.interpolate(cb, size=(ah, aw), mode="area")
        low = _morph_close_binary(low, int(close_radius))
        comp0 = _largest_component_lowres(low, threshold=float(component_threshold))
        if float(comp0.mean()) < float(min_area_fraction):
            out_components.append(torch.zeros_like(comp0))
            continue

        # Estimate the dominant color from confident pixels belonging to the
        # preliminary component.  This removes smooth foreground islands that
        # touch/approach the background in the flatness topology.
        full0 = F.interpolate(comp0, size=(h, w), mode="nearest")
        sel = (full0[0, 0] > 0.5) & (cb[0, 0] >= float(component_threshold))
        if int(sel.sum()) < 32:
            out_components.append(torch.zeros_like(comp0))
            continue
        my = torch.median(ys[bi, 0][sel])
        mcb = torch.median(cs[bi, 0][sel])
        mcr = torch.median(cs[bi, 1][sel])
        dy = (ys[bi:bi+1, 0:1] - my).abs()
        dc = torch.sqrt(
            (cs[bi:bi+1, 0:1] - mcb).square()
            + (cs[bi:bi+1, 1:2] - mcr).square()
            + 1e-12
        )
        gy = 1.0 - _smoothstep(
            dy, float(luma_tolerance) * 0.65, float(luma_tolerance)
        )
        gc = 1.0 - _smoothstep(
            dc, float(chroma_tolerance) * 0.65, float(chroma_tolerance)
        )
        refined = (cb * gy * gc).clamp(0.0, 1.0)

        low2 = F.interpolate(refined, size=(ah, aw), mode="area")
        low2 = _morph_close_binary(low2, int(close_radius))
        comp = _largest_component_lowres(low2, threshold=float(component_threshold))
        if float(comp.mean()) < float(min_area_fraction):
            comp = torch.zeros_like(comp)
        out_components.append(comp)

    low_component = torch.cat(out_components, dim=0)
    region = F.interpolate(low_component, size=(h, w), mode="bilinear", align_corners=False)
    if feather_sigma > 0:
        region = _smooth_axis(_smooth_axis(region, float(feather_sigma), "x"), float(feather_sigma), "y")
    return region.clamp(0.0, 1.0)


def make_dominant_large_surface_mask(
    y: torch.Tensor,
    c: torch.Tensor,
    *,
    luma_gate: torch.Tensor,
    raw_tone_veto: torch.Tensor,
    edge_support: torch.Tensor,
    preblur_sigma: float = 4.0,
    analysis_size: int = 256,
    component_threshold: float = 0.30,
    close_radius: int = 1,
    min_area_fraction: float = 0.08,
    chroma_tolerance: float = 0.080,
    luma_tolerance: float = 0.24,
    row_edge_barrier: float = 0.030,
    row_edge_guard_px: int = 3,
    feather_sigma: float = 5.0,
    coherence_sigma_x: float = 24.0,
    flat_full: float = 0.007,
    flat_none: float = 0.025,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find one dominant, large, close-colored/simple surface.

    Fine texture is suppressed before segmentation. Strong scene edges split
    candidate surfaces.  The largest connected component is selected at low
    resolution and then feathered at full resolution.

    Returns (soft_region, raw_candidate).
    """
    h, w = y.shape[-2:]
    # Strong preblur makes rough plaster / grain look like the underlying surface.
    surface = make_flat_mask(
        y, c,
        coherence_sigma_x=coherence_sigma_x,
        flat_full=max(1e-5, flat_full * 0.35),
        flat_none=max(flat_full * 0.40, flat_none * 0.35),
        preblur_sigma=preblur_sigma,
    )
    candidate = (surface * luma_gate * raw_tone_veto * edge_support).clamp(0.0, 1.0)

    # Dominant-color constraint: the rough-surface detector is intentionally
    # permissive, so without this step a wall can become connected to furniture
    # or fabric after downsampling. The largest surface is expected to dominate
    # the candidate population; robust medians provide its coarse Y/CbCr center.
    ys = _smooth_axis(_smooth_axis(y.float(), max(1.0, preblur_sigma), "x"), max(1.0, preblur_sigma), "y")
    cs = _smooth_axis(_smooth_axis(c.float(), max(1.0, preblur_sigma), "x"), max(1.0, preblur_sigma), "y")
    color_gate = torch.ones_like(candidate)
    for bi in range(y.shape[0]):
        sel = candidate[bi, 0] > 0.35
        if int(sel.sum()) < 32:
            sel = torch.ones_like(sel, dtype=torch.bool)
        my = torch.median(ys[bi, 0][sel])
        mcb = torch.median(cs[bi, 0][sel])
        mcr = torch.median(cs[bi, 1][sel])
        dy = (ys[bi:bi+1, 0:1] - my).abs()
        dc = torch.sqrt((cs[bi:bi+1, 0:1] - mcb).square() + (cs[bi:bi+1, 1:2] - mcr).square() + 1e-12)
        gy = 1.0 - _smoothstep(dy, float(luma_tolerance) * 0.65, float(luma_tolerance))
        gc = 1.0 - _smoothstep(dc, float(chroma_tolerance) * 0.65, float(chroma_tolerance))
        color_gate[bi:bi+1] = (gy * gc).clamp(0.0, 1.0)
    candidate = (candidate * color_gate).clamp(0.0, 1.0)

    # A strong boundary that spans a substantial fraction of a row should split
    # surfaces even if similar colors reconnect around the left/right image edge.
    # This is especially useful for a large wall above furniture/headboards.
    if row_edge_barrier > 0:
        row_edge = (1.0 - edge_support).mean(dim=-1, keepdim=True)
        rb = _smoothstep(row_edge, float(row_edge_barrier) * 0.80, float(row_edge_barrier))
        if row_edge_guard_px > 0:
            g = int(row_edge_guard_px)
            rb = F.max_pool2d(rb, kernel_size=(2*g+1, 1), stride=1, padding=(g, 0))
        candidate = candidate * (1.0 - rb).clamp(0.0, 1.0)

    scale = min(1.0, float(max(32, analysis_size)) / max(h, w))
    ah, aw = max(24, int(round(h * scale))), max(24, int(round(w * scale)))
    low = F.interpolate(candidate, size=(ah, aw), mode="area")
    low = _morph_close_binary(low, int(close_radius))
    comp = _largest_component_lowres(low, threshold=component_threshold)
    area = float(comp.mean())
    if area < float(min_area_fraction):
        return torch.zeros_like(candidate), candidate
    region = F.interpolate(comp, size=(h, w), mode="bilinear", align_corners=False)
    # Recover a smooth but edge-respecting visible region. The component defines
    # topology; tone eligibility defines where correction may actually appear.
    if feather_sigma > 0:
        region = _smooth_axis(_smooth_axis(region, feather_sigma, "x"), feather_sigma, "y")
    region = (region.clamp(0.0, 1.0) * luma_gate * raw_tone_veto).clamp(0.0, 1.0)
    return region, candidate


def _weighted_poly_baseline(row: torch.Tensor, weight: torch.Tensor, degree: int = 2, ridge: float = 1e-5) -> torch.Tensor:
    """Weighted low-order polynomial baseline along Y for BxCxHx1 rows."""
    if row.ndim != 4 or row.shape[-1] != 1:
        raise ValueError("row must be BxCxHx1")
    b, ch, h, _ = row.shape
    degree = int(max(0, min(int(degree), 5)))
    t = torch.linspace(-1.0, 1.0, h, device=row.device, dtype=row.dtype)
    X = torch.stack([t.pow(k) for k in range(degree + 1)], dim=1)  # HxP
    outs = []
    for bi in range(b):
        ch_out = []
        for ci in range(ch):
            w = weight[bi, 0, :, 0].float().clamp_min(0.0)
            yy = row[bi, ci, :, 0].float()
            sw = torch.sqrt(w + 1e-8)
            A = X.float() * sw[:, None]
            z = yy * sw
            ATA = A.T @ A + float(ridge) * torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
            ATz = A.T @ z
            coef = torch.linalg.solve(ATA, ATz)
            ch_out.append((X.float() @ coef).to(row.dtype))
        outs.append(torch.stack(ch_out, dim=0))
    return torch.stack(outs, dim=0).unsqueeze(-1)


def _large_surface_row_equalizer(
    y: torch.Tensor,
    c: torch.Tensor,
    *,
    region: torch.Tensor,
    edge_support: torch.Tensor,
    luma_gate: torch.Tensor,
    raw_tone_support: torch.Tensor,
    raw_tone_veto: torch.Tensor,
    luma_strength: float = 1.0,
    chroma_strength: float = 1.0,
    poly_degree: int = 2,
    row_sigma: float = 2.0,
    huber_k: float = 2.5,
    min_coverage: float = 0.04,
    eps: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equalize very broad row residuals on one dominant large surface.

    The equalizer is intentionally non-periodic. It robustly measures the
    dominant surface per row, fits a low-order illumination/color baseline, and
    treats departures from that baseline as residual rolling flicker. Only a
    row correction is applied; image texture itself is never blurred.
    """
    if luma_strength <= 0 and chroma_strength <= 0:
        z1 = torch.zeros_like(y)
        z2 = torch.zeros_like(c)
        return y, c, 0.0, 0.0, z1, z2, torch.zeros_like(region)
    support = (region * edge_support * _smoothstep(luma_gate, 0.15, 0.80) * _smoothstep(raw_tone_support, 0.05, 0.40)).clamp(0.0, 1.0)
    apply = (region * luma_gate * raw_tone_veto).clamp(0.0, 1.0)

    logy = torch.log(y.float().clamp_min(0.0) + float(eps))
    row_y, valid_y, conf_y, _ = _robust_row_consensus(logy, support, huber_k=huber_k, min_coverage=min_coverage)
    row_c, valid_c, conf_c, _ = _robust_row_consensus(c.float(), support, huber_k=huber_k, min_coverage=min_coverage)
    # Small row smoothing removes consensus noise, not image texture.
    if row_sigma > 0:
        row_y = _normalized_smooth_axis(row_y, valid_y, row_sigma, "y")
        row_c = _normalized_smooth_axis(row_c, valid_c, row_sigma, "y")

    cov = support.mean(dim=-1, keepdim=True)
    fit_weight = (cov * torch.minimum(conf_y, conf_c)).clamp(0.0, 1.0)
    baseline_y = _weighted_poly_baseline(row_y, fit_weight, degree=poly_degree)
    baseline_c = _weighted_poly_baseline(row_c, fit_weight, degree=poly_degree)
    corr_y = baseline_y - row_y
    corr_c = baseline_c - row_c

    corr_y_map = corr_y.expand(-1, 1, -1, y.shape[-1])
    corr_c_map = corr_c.expand(-1, 2, -1, c.shape[-1])
    apply_c = apply.expand(-1, 2, -1, -1)
    # Preserve mean exposure / surface color; only remove spatial row variation.
    my = (corr_y_map * apply).sum(dim=(-2, -1), keepdim=True) / apply.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    mc = (corr_c_map * apply_c).sum(dim=(-2, -1), keepdim=True) / apply_c.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    corr_y = corr_y - my
    corr_c = corr_c - mc

    out_y = (y.float() + float(eps)) * torch.exp(apply * corr_y * float(luma_strength)) - float(eps)
    out_c = c.float() + apply_c * corr_c * float(chroma_strength)
    yrms = float((apply * corr_y * float(luma_strength)).square().mean().sqrt())
    crms = float((apply_c * corr_c * float(chroma_strength)).square().mean().sqrt())
    return out_y, out_c, yrms, crms, corr_y, corr_c, apply

def _zero_weighted_mean(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.expand(-1, delta.shape[1], -1, -1)
    mean = (delta * w).sum(dim=(-2, -1), keepdim=True) / w.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return delta - mean


def _fill_row_profile(row: torch.Tensor, valid: torch.Tensor, sigma: float) -> torch.Tensor:
    return _normalized_smooth_axis(row, valid, max(1.0, sigma), "y")


def _residual_row_profile_correction(
    y: torch.Tensor,
    c: torch.Tensor,
    *,
    support: torch.Tensor,
    application_gate: torch.Tensor,
    edge_support: torch.Tensor,
    high_strength_guard: torch.Tensor | None = None,
    period: float,
    period_confidence: float,
    luma_strength: float,
    chroma_strength: float,
    narrow_ratio: float = 0.035,
    base_ratio: float = 0.40,
    eps: float = 0.02,
    huber_k: float = 2.5,
    min_coverage: float = 0.08,
    adaptive: bool = True,
    adaptive_x_ratio: float = 0.35,
    adaptive_y_ratio: float = 1.50,
    adaptive_corr_low: float = 0.15,
    adaptive_corr_high: float = 0.45,
    adaptive_max_gain: float = 1.00,
    no_harm: bool = True,
    surface_handoff: torch.Tensor | None = None,
    surface_luma_authority: float = 1.0,
    surface_chroma_authority: float = 1.0,
    surface_strength_cap: float = 0.75,
) -> tuple[
    torch.Tensor, torch.Tensor, float, float,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Texture-tolerant residual row-profile suppression with local amplitude.

    Period, phase and waveform are still estimated globally from robust row
    consensus.  When ``adaptive`` is enabled, only the waveform amplitude is
    fitted locally from covariance with the observed residual.  This preserves
    the physical row-coherent model while avoiding one global strength being
    forced onto unrelated foreground/background surfaces.
    """
    if luma_strength <= 0 and chroma_strength <= 0:
        z_y = torch.zeros_like(y)
        z_c = torch.zeros_like(c)
        z_m = torch.zeros_like(y)
        o_m = torch.ones_like(y)
        return y, c, 0.0, 0.0, z_y, z_c, z_m, z_m, o_m, o_m, z_m, z_m

    logy = torch.log(y.float().clamp_min(0.0) + float(eps))
    logy_res = _remove_linear_y_trend(logy, support)
    c_res = _remove_linear_y_trend(c.float(), support)

    row_y, valid_y, conf_y, _ = _robust_row_consensus(
        logy_res, support, huber_k=huber_k, min_coverage=min_coverage
    )
    row_c, valid_c, conf_c, _ = _robust_row_consensus(
        c_res, support, huber_k=huber_k, min_coverage=min_coverage
    )
    fill_sigma = max(1.0, period * 0.06)
    row_y = _fill_row_profile(row_y, valid_y, fill_sigma)
    row_c = _fill_row_profile(row_c, valid_c, fill_sigma)

    narrow_sigma = max(0.75, period * float(narrow_ratio))
    base_sigma = max(narrow_sigma + 0.5, period * float(base_ratio))
    narrow_y = _smooth_axis(row_y, narrow_sigma, "y")
    base_y = _smooth_axis(row_y, base_sigma, "y")
    narrow_c = _smooth_axis(row_c, narrow_sigma, "y")
    base_c = _smooth_axis(row_c, base_sigma, "y")
    corr_log_y = base_y - narrow_y
    corr_c = base_c - narrow_c

    period_weight = 0.85 + 0.15 * float(max(0.0, min(1.0, period_confidence)))
    row_conf_y = (conf_y * period_weight).clamp(0.0, 1.0)
    row_conf_c = (conf_c * period_weight).clamp(0.0, 1.0)
    apply_y = (application_gate * row_conf_y).clamp(0.0, 1.0)
    apply_c_scalar = (application_gate * row_conf_c).clamp(0.0, 1.0)
    apply_c = apply_c_scalar.expand(-1, 2, -1, -1)

    # Preserve the historical zero-DC behavior before fitting local amplitude.
    corr_y_map = corr_log_y.expand(-1, 1, -1, y.shape[-1])
    mean_y = (corr_y_map * apply_y).sum(dim=(-2, -1), keepdim=True) / apply_y.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    corr_log_y = corr_log_y - mean_y
    corr_c_map = corr_c.expand(-1, 2, -1, c.shape[-1])
    mean_c = (corr_c_map * apply_c).sum(dim=(-2, -1), keepdim=True) / apply_c.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    corr_c = corr_c - mean_c

    corr_y_map = corr_log_y.expand(-1, 1, -1, y.shape[-1])
    corr_c_map = corr_c.expand(-1, 2, -1, c.shape[-1])
    if adaptive:
        amp_y, evidence_y = _adaptive_profile_amplitude(
            logy_res, corr_y_map, support, edge_support,
            period=period,
            sigma_x_ratio=adaptive_x_ratio,
            sigma_y_ratio=adaptive_y_ratio,
            corr_low=adaptive_corr_low,
            corr_high=adaptive_corr_high,
            max_gain=adaptive_max_gain,
            narrow_ratio=narrow_ratio,
            base_ratio=base_ratio,
        )
        amp_c, evidence_c = _adaptive_profile_amplitude(
            c_res, corr_c_map, support, edge_support,
            period=period,
            sigma_x_ratio=adaptive_x_ratio,
            sigma_y_ratio=adaptive_y_ratio,
            corr_low=adaptive_corr_low,
            corr_high=adaptive_corr_high,
            max_gain=adaptive_max_gain,
            narrow_ratio=narrow_ratio,
            base_ratio=base_ratio,
        )
    else:
        amp_y = torch.ones_like(y)
        amp_c = torch.ones_like(y)
        evidence_y = torch.ones_like(y)
        evidence_c = torch.ones_like(y)

    field_y = corr_y_map * amp_y
    field_c = corr_c_map * amp_c.expand(-1, 2, -1, -1)

    # Varying amplitude can reintroduce a DC component. Remove it from the
    # *actual* adaptive field so profile cleanup still does not shift exposure
    # or global chroma merely because foreground/background gains differ.
    field_y = field_y - (field_y * apply_y).sum(dim=(-2, -1), keepdim=True) / apply_y.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    field_c = field_c - (field_c * apply_c).sum(dim=(-2, -1), keepdim=True) / apply_c.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

    if no_harm:
        proposed_y = apply_y * field_y * float(luma_strength)
        proposed_c = apply_c * field_c * float(chroma_strength)
        safe_y = _profile_no_harm_gate(
            logy_res, proposed_y, support, edge_support,
            period=period, narrow_ratio=narrow_ratio, base_ratio=base_ratio,
        )
        safe_c = _profile_no_harm_gate(
            c_res, proposed_c, support, edge_support,
            period=period, narrow_ratio=narrow_ratio, base_ratio=base_ratio,
        )
        apply_y = (apply_y * safe_y).clamp(0.0, 1.0)
        apply_c_scalar = (apply_c_scalar * safe_c).clamp(0.0, 1.0)
        apply_c = apply_c_scalar.expand(-1, 2, -1, -1)
        # Fold the safety result into the existing evidence debug maps rather
        # than creating another public statistic/API surface.
        evidence_y = (evidence_y * safe_y).clamp(0.0, 1.0)
        evidence_c = (evidence_c * safe_c).clamp(0.0, 1.0)

    # Bright soft-edge safeguard for intentionally aggressive profile strengths.
    # White flowers, masks and pale defocused boundaries can sit outside the
    # dominant-wall owner but still have almost no reliable local evidence for
    # a 1.8-2.0x row correction.  Limit only high-strength use near such bright
    # scene boundaries; darker skin/face interiors remain untouched.
    if high_strength_guard is not None:
        guard = high_strength_guard.float().clamp(0.0, 1.0)
        if guard.ndim != 4 or guard.shape[1] != 1 or guard.shape[-2:] != y.shape[-2:]:
            raise ValueError("high_strength_guard must be Bx1xHxW")
        bright_edge = ((1.0 - guard) * _smoothstep(y.float(), 0.62, 0.82)).clamp(0.0, 1.0)
        edge_cap = 0.50
        if float(luma_strength) > 1.0:
            ratio = edge_cap / max(float(luma_strength), 1e-6)
            scale = 1.0 - bright_edge * (1.0 - ratio)
            apply_y = (apply_y * scale).clamp(0.0, 1.0)
        if float(chroma_strength) > 1.0:
            ratio = edge_cap / max(float(chroma_strength), 1e-6)
            scale = 1.0 - bright_edge * (1.0 - ratio)
            apply_c_scalar = (apply_c_scalar * scale).clamp(0.0, 1.0)
            apply_c = apply_c_scalar.expand(-1, 2, -1, -1)

    # Final-stage handoff to the local flat cleaner.  Estimation, adaptive gain,
    # DC anchoring and no-harm validation above remain exactly as they were, so
    # foreground/profile behavior is unchanged.  Only the *visible* profile
    # contribution is capped on large smooth surfaces that the local flat stage
    # can safely own.  This prevents high profile strengths from painting broad
    # bright/dark lobes into defocused walls/backgrounds.
    if surface_handoff is not None and float(surface_strength_cap) >= 0.0:
        owner = surface_handoff.float().clamp(0.0, 1.0)
        if owner.ndim != 4 or owner.shape[1] != 1 or owner.shape[-2:] != y.shape[-2:]:
            raise ValueError("surface_handoff must be Bx1xHxW")
        cap = float(surface_strength_cap)

        # Extend the protected dominant surface a short distance *into* nearby
        # foreground boundaries, then release the cap smoothly.  This avoids a
        # visible profile-strength contour around heavily defocused silhouettes
        # without blurring the ownership mask across an entire face.  The wall
        # core is never weakened; only an outward transition skirt is added.
        short_side = min(owner.shape[-2:])
        skirt_radius = max(3, min(16, int(round(float(short_side) * 0.008))))
        skirt_sigma = max(2.0, min(12.0, float(short_side) * 0.008))
        hard_owner = (owner >= 0.50).to(owner.dtype)
        k = 2 * skirt_radius + 1
        dilated_owner = F.max_pool2d(
            hard_owner, kernel_size=k, stride=1, padding=skirt_radius
        )
        skirt = _smooth_axis(
            _smooth_axis(dilated_owner, skirt_sigma, "x"),
            skirt_sigma, "y",
        ).clamp(0.0, 1.0)
        transition_owner = torch.maximum(owner, skirt).clamp(0.0, 1.0)

        if float(luma_strength) > cap and float(surface_luma_authority) > 0.0:
            authority_y = float(max(0.0, min(1.0, surface_luma_authority)))
            ratio_y = cap / max(float(luma_strength), 1e-6)
            handoff_y = transition_owner if float(luma_strength) > 1.0 else owner
            scale_y = 1.0 - handoff_y * authority_y * (1.0 - ratio_y)
            apply_y = (apply_y * scale_y).clamp(0.0, 1.0)
        if float(chroma_strength) > cap and float(surface_chroma_authority) > 0.0:
            authority_c = float(max(0.0, min(1.0, surface_chroma_authority)))
            ratio_c = cap / max(float(chroma_strength), 1e-6)
            handoff_c = transition_owner if float(chroma_strength) > 1.0 else owner
            scale_c = 1.0 - handoff_c * authority_c * (1.0 - ratio_c)
            apply_c_scalar = (apply_c_scalar * scale_c).clamp(0.0, 1.0)
            apply_c = apply_c_scalar.expand(-1, 2, -1, -1)

    applied_log_y = apply_y * field_y * float(luma_strength)
    # Preserve highlight headroom when a strong positive residual profile lands
    # on bright foreground detail.  Near-white flowers, masks and specular
    # details can otherwise be pushed to clipping even though they are not part
    # of the dominant smooth background.  Negative corrections are untouched.
    # The limiter fades in only through the bright range and still allows a
    # substantial fraction of the remaining headroom.
    bright_gate = _smoothstep(y.float(), 0.70, 0.90)
    headroom_fraction = 1.0 - 0.45 * bright_gate  # 1.0 -> 0.55 toward highlights
    max_profile_y = y.float() + headroom_fraction * (1.0 - y.float())
    max_positive_log = torch.log(
        (max_profile_y + float(eps)).clamp_min(1e-6)
        / (y.float() + float(eps)).clamp_min(1e-6)
    )
    applied_log_y = torch.where(
        applied_log_y > 0.0,
        torch.minimum(applied_log_y, max_positive_log),
        applied_log_y,
    )

    out_y = (y.float() + float(eps)) * torch.exp(applied_log_y) - float(eps)
    out_c = c.float() + apply_c * field_c * float(chroma_strength)
    y_rms = float(applied_log_y.square().mean().sqrt())
    c_rms = float((apply_c * field_c * float(chroma_strength)).square().mean().sqrt())
    combined_conf = torch.minimum(row_conf_y, row_conf_c).clamp(0.0, 1.0)
    effective_apply = (apply_y * _smoothstep(amp_y, 0.05, 0.50)).clamp(0.0, 1.0)
    return (
        out_y, out_c, y_rms, c_rms,
        field_y, field_c, combined_conf, effective_apply,
        amp_y, amp_c, evidence_y, evidence_c,
    )

def apply_flat_region_filter(
    y: torch.Tensor,
    c: torch.Tensor,
    *,
    luma_field: torch.Tensor | None = None,
    chroma_delta_hint: torch.Tensor | None = None,
    band_period_px: float = 0.0,
    period_sigma_ratio: float = 0.25,
    coherence_sigma_x: float = 16.0,
    flat_full: float = 0.007,
    flat_none: float = 0.025,
    luma_strength: float = 0.70,
    chroma_strength: float = 0.85,
    luma_highpass: str | None = "#232323",
    luma_lowpass: str | None = "#efefef",
    shadow_ramp_lstar: float = 12.0,
    highlight_ramp_lstar: float = 8.0,
    luma_spatial_feather: float = 1.25,
    base_lstar_sigma_ratio: float = 0.40,
    edge_preblur_sigma: float = 1.5,
    edge_low: float = 0.018,
    edge_high: float = 0.055,
    edge_guard_px: int = 2,
    broad_structure_sigma: float = 8.0,
    broad_structure_low: float = 0.025,
    broad_structure_high: float = 0.080,
    broad_structure_guard_px: int = 8,
    broad_structure_feather: float = 6.0,
    coarse_preblur_sigma: float = 1.0,
    coarse_blend_weight: float = 0.25,
    support_low: float = 0.25,
    support_high: float = 0.75,
    blend_feather: float = 0.75,
    local_extent_sigma: float = 32.0,
    local_extent_low: float = 0.18,
    local_extent_high: float = 0.50,
    local_color_sigma: float = 44.0,
    local_color_preblur: float = 2.0,
    local_color_luma_tolerance: float = 7.0,
    local_color_chroma_tolerance: float = 0.030,
    local_fill_sigma: float = 36.0,
    local_fill_low: float = 0.38,
    local_fill_high: float = 0.68,
    local_edge_distance: int = 6,
    local_edge_feather: float = 3.0,
    local_correction_horizontal_sigma: float = 128.0,
    local_application_horizontal_sigma: float = 48.0,
    residual_profile_luma_strength: float = 0.0,
    residual_profile_chroma_strength: float = 0.0,
    residual_profile_narrow_ratio: float = 0.035,
    residual_profile_base_ratio: float = 0.40,
    residual_profile_band_period_px: float = 0.0,
    period_mode: str = "multiscale",
    period_min_px: float = 12.0,
    period_max_fraction: float = 0.60,
    period_analysis_h: int = 512,
    residual_profile_huber_k: float = 2.5,
    residual_profile_min_coverage: float = 0.08,
    residual_profile_adaptive: bool = True,
    residual_profile_adaptive_x_ratio: float = 0.35,
    residual_profile_adaptive_y_ratio: float = 1.50,
    residual_profile_adaptive_corr_low: float = 0.15,
    residual_profile_adaptive_corr_high: float = 0.45,
    residual_profile_adaptive_max_gain: float = 1.00,
    residual_profile_no_harm: bool = True,
    surface_equalizer_enabled: bool = False,
    surface_equalizer_luma_strength: float = 1.0,
    surface_equalizer_chroma_strength: float = 1.0,
    surface_equalizer_poly_degree: int = 2,
    surface_equalizer_preblur_sigma: float = 4.0,
    surface_equalizer_analysis_size: int = 256,
    surface_equalizer_component_threshold: float = 0.30,
    surface_equalizer_close_radius: int = 1,
    surface_equalizer_min_area_fraction: float = 0.08,
    surface_equalizer_chroma_tolerance: float = 0.080,
    surface_equalizer_luma_tolerance: float = 0.24,
    surface_equalizer_row_edge_barrier: float = 0.030,
    surface_equalizer_row_edge_guard_px: int = 3,
    surface_equalizer_feather_sigma: float = 5.0,
    surface_equalizer_row_sigma: float = 2.0,
    surface_equalizer_huber_k: float = 2.5,
    surface_equalizer_min_coverage: float = 0.04,
    local_user_mask: torch.Tensor | None = None,
    profile_user_mask: torch.Tensor | None = None,
    surface_user_mask: torch.Tensor | None = None,
    preserve_global_mean: bool = True,
    debug_out: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, FlatFilterStats]:
    """Flatten row-coherent residuals with edge-aware masked estimation."""
    if y.ndim != 4 or y.shape[1] != 1 or c.ndim != 4 or c.shape[1] != 2:
        raise ValueError("Expected Y Bx1xHxW and C Bx2xHxW")

    def _user_mask(mask: torch.Tensor | None, name: str) -> torch.Tensor | None:
        if mask is None:
            return None
        m=mask.to(device=y.device,dtype=y.dtype)
        if m.ndim==3:
            m=m.unsqueeze(1)
        if m.ndim!=4 or m.shape[1]!=1 or m.shape[0]!=y.shape[0] or m.shape[-2:]!=y.shape[-2:]:
            raise ValueError(f"{name} must be Bx1xHxW and match Y")
        return m.clamp(0.0,1.0)

    local_user_mask=_user_mask(local_user_mask,"local_user_mask")
    profile_user_mask=_user_mask(profile_user_mask,"profile_user_mask")
    surface_user_mask=_user_mask(surface_user_mask,"surface_user_mask")
    if period_sigma_ratio <= 0:
        raise ValueError("period_sigma_ratio must be > 0")
    if not (0.0 <= coarse_blend_weight <= 1.0):
        raise ValueError("coarse_blend_weight must be in [0,1]")
    if broad_structure_sigma < 0 or broad_structure_guard_px < 0 or broad_structure_feather < 0:
        raise ValueError("broad-structure sigma/guard/feather must be >= 0")
    if not (0.0 <= broad_structure_low < broad_structure_high):
        raise ValueError("broad-structure thresholds must satisfy 0 <= low < high")
    if local_correction_horizontal_sigma < 0:
        raise ValueError("local_correction_horizontal_sigma must be >= 0")
    if local_application_horizontal_sigma < 0:
        raise ValueError("local_application_horizontal_sigma must be >= 0")
    if residual_profile_adaptive_x_ratio <= 0 or residual_profile_adaptive_y_ratio <= 0:
        raise ValueError("adaptive profile x/y ratios must be > 0")
    if not (0.0 <= residual_profile_adaptive_corr_low < residual_profile_adaptive_corr_high <= 1.0):
        raise ValueError("adaptive profile correlation thresholds must satisfy 0 <= low < high <= 1")
    if residual_profile_adaptive_max_gain <= 0:
        raise ValueError("residual_profile_adaptive_max_gain must be > 0")

    h, _ = y.shape[-2:]

    # Scene edges and a relaxed raw-tone gate are available before band-period
    # estimation.  They define trustworthy evidence without requiring flatness.
    edge_support = make_scene_edge_support(
        y, c,
        preblur_sigma=edge_preblur_sigma,
        edge_low=edge_low,
        edge_high=edge_high,
        guard_px=edge_guard_px,
    )
    local_filter_enabled = bool(luma_strength > 0.0 or chroma_strength > 0.0)
    if local_filter_enabled:
        broad_structure_support = make_broad_structure_support(
            y, c,
            sigma_px=broad_structure_sigma,
            edge_low=broad_structure_low,
            edge_high=broad_structure_high,
            guard_px=broad_structure_guard_px,
            feather_sigma=broad_structure_feather,
        )
    else:
        # Residual-profile-only runs do not need the extra broad-edge analysis.
        broad_structure_support = torch.ones_like(y)
    local_edge_support = (edge_support * broad_structure_support).clamp(0.0, 1.0)
    raw_rgb = y_cbcr_to_rgb(y.float(), c.float()).clamp(0.0, 1.0)
    raw_lstar = rgb_to_lab_lstar(raw_rgb)
    raw_support_gate, _, _ = _gate_from_lstar(
        raw_lstar,
        highpass=luma_highpass,
        lowpass=luma_lowpass,
        shadow_ramp_lstar=max(18.0, shadow_ramp_lstar * 1.5),
        highlight_ramp_lstar=max(12.0, highlight_ramp_lstar * 1.5),
    )
    period_support = (edge_support * raw_support_gate).clamp(0.0, 1.0)

    # Preserve v1.5 behavior for the *local* flattener.  The new harmonic-aware
    # period is isolated to the optional global row-profile stage.
    if band_period_px > 0:
        local_period = float(band_period_px)
    else:
        local_period = float(estimate_band_period(luma_field, chroma_delta_hint, full_height=h))

    profile_enabled = residual_profile_luma_strength > 0 or residual_profile_chroma_strength > 0
    if residual_profile_band_period_px > 0:
        period_est = BandPeriodEstimate(
            float(residual_profile_band_period_px), 1.0, ((float(residual_profile_band_period_px), 1.0),)
        )
    elif band_period_px > 0:
        period_est = BandPeriodEstimate(float(band_period_px), 1.0, ((float(band_period_px), 1.0),))
    elif not profile_enabled:
        period_est = BandPeriodEstimate(local_period, 0.0, ((local_period, 0.0),))
    elif str(period_mode).lower() == "legacy":
        period_est = BandPeriodEstimate(local_period, 0.5, ((local_period, 0.5),))
    else:
        period_est = estimate_band_period_multiscale(
            luma_field, chroma_delta_hint,
            full_height=h, y=y, c=c, support=period_support,
            min_period_px=period_min_px,
            max_period_fraction=period_max_fraction,
            analysis_h=period_analysis_h,
            huber_k=residual_profile_huber_k,
        )
    profile_period = float(period_est.period_px)
    sigma_y = max(0.75, local_period * float(period_sigma_ratio))
    sigma_y = min(sigma_y, max(1.0, h / 10.0))

    base_lstar = make_band_resistant_base_lstar(
        y, c,
        band_period_px=local_period,
        edge_support=edge_support,
        highpass=luma_highpass,
        lowpass=luma_lowpass,
        shadow_ramp_lstar=shadow_ramp_lstar,
        highlight_ramp_lstar=highlight_ramp_lstar,
        sigma_ratio=base_lstar_sigma_ratio,
    )
    luma_gate, lstar_min, lstar_max = make_perceptual_luma_gate(
        y, c,
        highpass=luma_highpass,
        lowpass=luma_lowpass,
        shadow_ramp_lstar=shadow_ramp_lstar,
        highlight_ramp_lstar=highlight_ramp_lstar,
        spatial_feather=luma_spatial_feather,
        base_lstar=base_lstar,
    )
    # raw_support_gate was computed before period estimation and is reused here.


    fine_mask = make_flat_mask(
        y, c,
        coherence_sigma_x=coherence_sigma_x,
        flat_full=flat_full,
        flat_none=flat_none,
        preblur_sigma=0.0,
    )
    # Coarse segmentation sees through fine wall texture / grain. Thresholds
    # are tightened because pre-smoothing reduces gradient magnitude.
    coarse_mask = make_flat_mask(
        y, c,
        coherence_sigma_x=coherence_sigma_x,
        flat_full=max(1e-5, flat_full * 0.50),
        flat_none=max(flat_full * 0.55, flat_none * 0.50),
        preblur_sigma=coarse_preblur_sigma,
    )

    support_tone = _smoothstep(luma_gate, 0.15, 0.80)
    raw_tone_support = _smoothstep(raw_support_gate, 0.05, 0.40)
    # Much softer raw-tone veto for the visible blend: only truly excluded dark/
    # clipped pixels are blocked. Ordinary flicker troughs remain eligible via
    # the band-resistant base-L* gate.
    raw_tone_veto = _smoothstep(raw_support_gate, 0.01, 0.15)

    # Local-island guard: smooth skin / tiny object patches may look locally flat,
    # but they are not part of a sufficiently large surrounding surface.
    extent_candidate = (torch.maximum(fine_mask, coarse_mask) * luma_gate * raw_tone_veto).clamp(0.0, 1.0)
    local_extent_gate = make_large_surface_extent_gate(
        extent_candidate, sigma_px=local_extent_sigma,
        low_fraction=local_extent_low, high_fraction=local_extent_high,
    )
    local_color_gate, local_fill_gate, local_edge_distance_gate, local_surface_gate = make_local_same_surface_gates(
        y, c,
        base_lstar=base_lstar, candidate=extent_candidate, edge_support=edge_support,
        color_sigma_px=local_color_sigma, color_preblur_sigma=local_color_preblur,
        luma_tolerance_lstar=local_color_luma_tolerance, chroma_tolerance=local_color_chroma_tolerance,
        fill_sigma_px=local_fill_sigma, fill_low=local_fill_low, fill_high=local_fill_high,
        edge_distance_px=local_edge_distance, edge_feather_sigma=local_edge_feather,
    )
    local_safe_gate = (local_extent_gate * local_surface_gate).clamp(0.0, 1.0)

    # Broad structure-density guard. Long high-contrast fixtures, window frames,
    # moulding, etc. can leave fragmented flat support whose geometry mimics a
    # row/column correction. Suppress the visible local correction in a soft
    # neighborhood around dense scene edges while leaving open flat surfaces
    # untouched. This is especially important after vertical-band axis
    # normalization, where a ceiling light can otherwise seed a displayed
    # vertical streak.
    structure_density = (
        1.0 - _smooth_axis(_smooth_axis(edge_support, 16.0, "x"), 16.0, "y")
    ).clamp(0.0, 1.0)
    local_structure_gate = (1.0 - _smoothstep(structure_density, 0.10, 0.24)).clamp(0.0, 1.0)
    local_safe_gate = (local_safe_gate * local_structure_gate * broad_structure_support).clamp(0.0, 1.0)

    # Estimation support keeps the proven v1.9 extent/edge behavior. The new
    # same-surface safety layer is applied to the visible local correction only;
    # this prevents small skin/object islands without weakening the wall's
    # correction field itself.
    support_surface = _smoothstep(coarse_mask, support_low, support_high) * local_extent_gate
    support = (support_surface * support_tone * raw_tone_support * local_edge_support).clamp(0.0, 1.0)

    # Softer blend controls how much correction is shown. A small amount of the
    # coarse segmentation lets textured, close-colored walls participate without
    # turning the actual image into a blur.
    surface_blend = torch.maximum(fine_mask, coarse_mask * float(coarse_blend_weight))
    blend = (surface_blend * local_safe_gate * luma_gate * raw_tone_veto).clamp(0.0, 1.0)
    if blend_feather > 0:
        blend = _smooth_axis(_smooth_axis(blend, blend_feather, "x"), blend_feather, "y")

    # Residual profile runs before the local flat correction.  v0.38 applied the
    # local correction first, then fitted the adaptive profile to that modified
    # image.  This meant local-mask artifacts could become evidence for the
    # profile and get reinforced.  Keeping the global/profile stage first makes
    # its result independent of whether Flat-region cleanup is enabled.
    profile_support = (edge_support * support_tone * raw_tone_support).clamp(0.0, 1.0)
    profile_application_gate = (luma_gate * raw_tone_veto).clamp(0.0, 1.0)
    if blend_feather > 0:
        profile_application_gate = _smooth_axis(
            _smooth_axis(profile_application_gate, blend_feather, "x"), blend_feather, "y"
        )
    if profile_user_mask is not None:
        # User paint masks are final application gates. Apply them after all
        # automatic feathering so a painted boundary is never leaked across.
        profile_application_gate = (profile_application_gate * profile_user_mask).clamp(0.0,1.0)

    # Dominant-surface stage ownership.  The previous handoff blurred every
    # large/simple patch by 16 px.  On defocused scenes that blur could leak a
    # wall's ownership into a smooth foreground face and unintentionally cap the
    # very strong profile correction the user wanted on skin.  Select one large
    # connected, color-consistent surface instead and feather it only lightly.
    if local_filter_enabled:
        handoff_candidate = (
            torch.maximum(fine_mask, coarse_mask)
            * local_extent_gate * local_surface_gate * luma_gate * raw_tone_veto
        ).clamp(0.0, 1.0)
        profile_surface_handoff = make_profile_surface_handoff(
            y, c, handoff_candidate,
            analysis_size=256,
            component_threshold=0.55,
            close_radius=1,
            min_area_fraction=0.08,
            luma_tolerance=0.30,
            chroma_tolerance=0.06,
            feather_sigma=4.0,
        )
        if local_user_mask is not None:
            # A masked local stage only owns the area the user painted. It must
            # not cap Residual Profile elsewhere just because the automatic flat
            # detector found another smooth surface.
            profile_surface_handoff = (profile_surface_handoff * local_user_mask).clamp(0.0,1.0)
        # If the local stage has been intentionally weakened, reduce the handoff
        # proportionally rather than leaving smooth surfaces under-corrected.
        handoff_luma_authority = max(0.0, min(1.0, float(luma_strength) / 0.70))
        handoff_chroma_authority = max(0.0, min(1.0, float(chroma_strength) / 0.85))
    else:
        profile_surface_handoff = torch.zeros_like(y)
        handoff_luma_authority = 0.0
        handoff_chroma_authority = 0.0
    (
        profile_out_y, profile_out_c, py_rms, pc_rms, profile_y, profile_c,
        profile_confidence, profile_apply, profile_amp_y, profile_amp_c,
        profile_evidence_y, profile_evidence_c,
    ) = _residual_row_profile_correction(
        y,
        c,
        support=profile_support,
        application_gate=profile_application_gate,
        edge_support=edge_support,
        high_strength_guard=torch.minimum(edge_support, broad_structure_support),
        period=profile_period,
        period_confidence=period_est.confidence,
        luma_strength=residual_profile_luma_strength,
        chroma_strength=residual_profile_chroma_strength,
        narrow_ratio=residual_profile_narrow_ratio,
        base_ratio=residual_profile_base_ratio,
        huber_k=residual_profile_huber_k,
        min_coverage=residual_profile_min_coverage,
        adaptive=residual_profile_adaptive,
        adaptive_x_ratio=residual_profile_adaptive_x_ratio,
        adaptive_y_ratio=residual_profile_adaptive_y_ratio,
        adaptive_corr_low=residual_profile_adaptive_corr_low,
        adaptive_corr_high=residual_profile_adaptive_corr_high,
        adaptive_max_gain=residual_profile_adaptive_max_gain,
        no_harm=residual_profile_no_harm,
        surface_handoff=profile_surface_handoff,
        surface_luma_authority=handoff_luma_authority,
        surface_chroma_authority=handoff_chroma_authority,
        surface_strength_cap=0.50,
    )

    # Local flat-region cleanup is now a residual stage after the profile.  Its
    # safety/support masks are still derived from the original image, but the
    # correction itself is estimated from the profile-corrected image so the two
    # stages do not both chase the same already-corrected bands.
    coh_y = _normalized_smooth_axis(profile_out_y.float(), support, coherence_sigma_x, "x")
    coh_c = _normalized_smooth_axis(profile_out_c.float(), support, coherence_sigma_x, "x")
    # Preserve real horizontal scene structure while estimating the residual
    # vertical band correction.  A masked Gaussian can bridge across excluded
    # edge rows; the edge-aware recursive smoother instead attenuates evidence
    # propagation at those boundaries without creating hard segmented seams.
    smooth_y = _edge_aware_normalized_smooth_y(coh_y, support, local_edge_support, sigma_y)
    smooth_c = _edge_aware_normalized_smooth_y(coh_c, support, local_edge_support, sigma_y)
    dy = smooth_y - coh_y
    dc = smooth_c - coh_c
    if local_correction_horizontal_sigma > 0:
        # Once a trustworthy local correction has been estimated, regularize the
        # correction field itself without re-applying the fragmented support mask.
        # The visible blend still protects objects/edges.  Using normalized
        # smoothing with ``support`` here (v0.38) preserved support-shaped islands
        # on large white backgrounds and could turn them into patchy corrections.
        dy = _smooth_axis(dy, float(local_correction_horizontal_sigma), "x")
        dc = _smooth_axis(dc, float(local_correction_horizontal_sigma), "x")

    effective_blend = blend if local_user_mask is None else (blend * local_user_mask).clamp(0.0,1.0)
    if preserve_global_mean:
        dy = _zero_weighted_mean(dy, effective_blend)
        dc = _zero_weighted_mean(dc, effective_blend)

    # The visible blend mask contains holes around people/objects by design.
    # Multiplying a row-varying correction by that fragmented mask can itself
    # draw object silhouettes into the result: the background is corrected while
    # the protected object is not, so every residual bright/dark row becomes an
    # edge-local halo.  Regularize the *applied correction field* across X after
    # blending.  This softly bridges small/medium protected gaps without blurring
    # the source image or letting excluded pixels participate in estimation.
    local_y_field = effective_blend * dy
    local_c_field = effective_blend.expand(-1, 2, -1, -1) * dc
    if local_application_horizontal_sigma > 0:
        local_y_field = _smooth_axis(local_y_field, float(local_application_horizontal_sigma), "x")
        local_c_field = _smooth_axis(local_c_field, float(local_application_horizontal_sigma), "x")
    if local_user_mask is not None:
        # Horizontal regularization may bridge across the mask edge; trim the
        # final field again so the stage is strictly confined to painted alpha.
        local_y_field = local_y_field * local_user_mask
        local_c_field = local_c_field * local_user_mask.expand(-1,2,-1,-1)

    out_y = profile_out_y.float() + local_y_field * float(luma_strength)
    out_c = profile_out_c.float() + local_c_field * float(chroma_strength)

    # Optional v1.7 large-surface equalizer for extremely broad/few-cycle
    # residuals that are intentionally treated as illumination by the periodic
    # profile stage. It selects one dominant large simple surface and fits only
    # a low-order vertical baseline, so image texture itself is untouched.
    surface_region = torch.zeros_like(y)
    surface_candidate = torch.zeros_like(y)
    eq_y_rms = 0.0
    eq_c_rms = 0.0
    equalizer_y = torch.zeros_like(y)
    equalizer_c = torch.zeros_like(c)
    equalizer_apply = torch.zeros_like(y)
    if surface_equalizer_enabled:
        surface_region, surface_candidate = make_dominant_large_surface_mask(
            y, c,
            luma_gate=luma_gate, raw_tone_veto=raw_tone_veto, edge_support=edge_support,
            preblur_sigma=surface_equalizer_preblur_sigma,
            analysis_size=surface_equalizer_analysis_size,
            component_threshold=surface_equalizer_component_threshold,
            close_radius=surface_equalizer_close_radius,
            min_area_fraction=surface_equalizer_min_area_fraction,
            chroma_tolerance=surface_equalizer_chroma_tolerance,
            luma_tolerance=surface_equalizer_luma_tolerance,
            row_edge_barrier=surface_equalizer_row_edge_barrier,
            row_edge_guard_px=surface_equalizer_row_edge_guard_px,
            feather_sigma=surface_equalizer_feather_sigma,
            coherence_sigma_x=max(coherence_sigma_x, 24.0),
            flat_full=flat_full, flat_none=flat_none,
        )
        before_eq_y, before_eq_c = out_y, out_c
        eq_out_y, eq_out_c, eq_y_rms, eq_c_rms, equalizer_y, equalizer_c, equalizer_apply = _large_surface_row_equalizer(
            out_y, out_c, region=surface_region, edge_support=edge_support, luma_gate=luma_gate,
            raw_tone_support=raw_support_gate, raw_tone_veto=raw_tone_veto,
            luma_strength=surface_equalizer_luma_strength, chroma_strength=surface_equalizer_chroma_strength,
            poly_degree=surface_equalizer_poly_degree, row_sigma=surface_equalizer_row_sigma,
            huber_k=surface_equalizer_huber_k, min_coverage=surface_equalizer_min_coverage,
        )
        if surface_user_mask is not None:
            # Keep the equalizer's robust large-surface fit unchanged, then gate
            # only its visible delta. This gives a predictable paint mask even
            # when the user paints only a small part of the dominant surface.
            out_y = before_eq_y + surface_user_mask * (eq_out_y - before_eq_y)
            out_c = before_eq_c + surface_user_mask.expand(-1,2,-1,-1) * (eq_out_c - before_eq_c)
            equalizer_apply = equalizer_apply * surface_user_mask
            eq_y_rms = float((out_y - before_eq_y).square().mean().sqrt())
            eq_c_rms = float((out_c - before_eq_c).square().mean().sqrt())
        else:
            out_y, out_c = eq_out_y, eq_out_c

    if debug_out is not None:
        debug_out.clear()
        debug_out.update({
            "fine_mask": fine_mask.detach(),
            "coarse_mask": coarse_mask.detach(),
            "local_extent_gate": local_extent_gate.detach(),
            "local_color_gate": local_color_gate.detach(),
            "local_fill_gate": local_fill_gate.detach(),
            "local_edge_distance_gate": local_edge_distance_gate.detach(),
            "local_surface_gate": local_surface_gate.detach(),
            "local_safe_gate": local_safe_gate.detach(),
            "local_structure_gate": local_structure_gate.detach(),
            "support_mask": support.detach(),
            "blend_mask": effective_blend.detach(),
            "user_mask_flat": (local_user_mask if local_user_mask is not None else torch.ones_like(y)).detach(),
            "user_mask_profile": (profile_user_mask if profile_user_mask is not None else torch.ones_like(y)).detach(),
            "user_mask_broad": (surface_user_mask if surface_user_mask is not None else torch.ones_like(y)).detach(),
            "local_applied_y": local_y_field.detach(),
            "local_applied_c": local_c_field.detach(),
            "profile_support_mask": profile_support.detach(),
            "profile_application_gate": profile_application_gate.detach(),
            "profile_surface_handoff": profile_surface_handoff.detach(),
            "profile_confidence": profile_confidence.detach(),
            "profile_apply_mask": profile_apply.detach(),
            "profile_adaptive_gain_y": profile_amp_y.detach(),
            "profile_adaptive_gain_c": profile_amp_c.detach(),
            "profile_adaptive_evidence_y": profile_evidence_y.detach(),
            "profile_adaptive_evidence_c": profile_evidence_c.detach(),
            "period_support": period_support.detach(),
            "edge_support": edge_support.detach(),
            "broad_structure_support": broad_structure_support.detach(),
            "luma_gate": luma_gate.detach(),
            "raw_tone_support": raw_tone_support.detach(),
            "raw_tone_veto": raw_tone_veto.detach(),
            "base_lstar": (base_lstar / 100.0).clamp(0.0, 1.0).detach(),
            "profile_y": profile_y.detach(),
            "profile_c": profile_c.detach(),
            "surface_equalizer_candidate": surface_candidate.detach(),
            "surface_equalizer_region": surface_region.detach(),
            "surface_equalizer_apply": equalizer_apply.detach(),
            "surface_equalizer_y": equalizer_y.detach(),
            "surface_equalizer_c": equalizer_c.detach(),
        })

    with torch.no_grad():
        stats = FlatFilterStats(
            flat_fraction=float((effective_blend > 0.5).float().mean()),
            support_fraction=float((support > 0.5).float().mean()),
            coarse_fraction=float((coarse_mask > 0.5).float().mean()),
            edge_support_fraction=float((edge_support > 0.5).float().mean()),
            luma_gate_fraction=float((luma_gate > 0.5).float().mean()),
            luma_min_lstar=lstar_min,
            luma_max_lstar=lstar_max,
            band_period_px=float(local_period),
            sigma_y_px=float(sigma_y),
            y_delta_rms=float(local_y_field.square().mean().sqrt()),
            c_delta_rms=float(local_c_field.square().mean().sqrt()),
            profile_y_rms=py_rms,
            profile_c_rms=pc_rms,
            profile_period_px=float(profile_period),
            band_confidence=float(period_est.confidence),
            profile_support_fraction=float((profile_support > 0.5).float().mean()),
            profile_apply_fraction=float((profile_apply > 0.5).float().mean()),
            profile_confidence_mean=float(profile_confidence.mean()),
            local_extent_fraction=float((local_extent_gate > 0.5).float().mean()),
            local_color_fraction=float((local_color_gate > 0.5).float().mean()),
            local_fill_fraction=float((local_fill_gate > 0.5).float().mean()),
            local_edge_distance_fraction=float((local_edge_distance_gate > 0.5).float().mean()),
            local_surface_fraction=float((local_safe_gate > 0.5).float().mean()),
            surface_equalizer_fraction=float((equalizer_apply > 0.5).float().mean()),
            surface_equalizer_y_rms=float(eq_y_rms),
            surface_equalizer_c_rms=float(eq_c_rms),
            band_candidates=period_est.candidates,
        )
    return out_y, out_c, effective_blend, stats
