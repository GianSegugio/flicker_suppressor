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

The optional residual-profile stage detects one or more straight-axis PWM timing
families and estimates their cleanup on radiometrically coherent image regions.
A global/cycle-consensus model remains the conservative baseline; a segmented
held-out fit may change the local mixture, phase and harmonic balance only when
it predicts unseen X samples better.  This also allows several LED sources with
different PWM periods/phases to coexist without treating every visible stripe as
an independent free parameter.  Broad residual cleanup remains a separate opt-in
stage and can use either the original dominant-surface equalizer or a multi-surface
Y/CbCr consensus.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from color_space import y_cbcr_to_rgb
from pwm_timing import refine_period_phase_slope, coherent_mode_power


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
    broad_consensus_confidence: float
    broad_consensus_regions: int
    band_candidates: tuple[tuple[float, float], ...]
    pwm_periods: tuple[float, ...]
    pwm_polish_y_rms: float = 0.0
    pwm_polish_c_rms: float = 0.0
    pwm_polish_passes: int = 0
    pwm_polish_improvement: float = 0.0


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



def _pwm_profile_no_harm_gate(
    residual: torch.Tensor,
    applied_delta: torch.Tensor,
    support: torch.Tensor,
    edge_support: torch.Tensor,
    *,
    period: float,
    base_ratio: float,
) -> torch.Tensor:
    """X-only no-harm validator for an edge-locked PWM proposal.

    Total high-frequency energy is a poor validator for concert/event images:
    texture, haze, hair and JPEG noise can dwarf a small but coherent PWM stripe.
    Instead measure only the residual component *along the proposed PWM field*.
    A proposal is accepted where adding it reduces that template projection.
    The resulting gate varies only across X, so it cannot trace an individual
    stripe or object boundary.
    """
    if residual.shape != applied_delta.shape:
        raise ValueError("PWM no-harm residual and applied_delta must match")
    if support.ndim != 4 or support.shape[1] != 1 or support.shape[-2:] != residual.shape[-2:]:
        raise ValueError("PWM no-harm support must be Bx1xHxW")

    base_sigma = max(12.0, float(period) * max(0.75, float(base_ratio) * 2.0))
    r = residual.float() - _smooth_axis(residual.float(), base_sigma, "y")
    q = applied_delta.float() - _smooth_axis(applied_delta.float(), base_sigma, "y")

    w = (support.float().clamp(0.0, 1.0) * edge_support.float().clamp(0.0, 1.0)).clamp(0.0, 1.0)
    wc = w.expand(-1, residual.shape[1], -1, -1)
    num = (r * q * wc).sum(dim=1, keepdim=True).sum(dim=-2, keepdim=True)
    den = (q.square() * wc).sum(dim=1, keepdim=True).sum(dim=-2, keepdim=True)
    # After applying q: dot(r+q, q) = num + den.
    before = num.square() / den.clamp_min(1e-12)
    after = (num + den).square() / den.clamp_min(1e-12)
    coverage = w.mean(dim=-2, keepdim=True)

    sx = max(48.0, min(256.0, float(period)))
    before = _smooth_axis(before, sx, "x")
    after = _smooth_axis(after, sx, "x")
    coverage = _smooth_axis(coverage, sx, "x")
    signed = _smooth_axis(num / den.clamp_min(1e-12), sx, "x")

    improvement = (before - after) / before.clamp_min(1e-12)
    # A correct correction points opposite to the residual, hence negative
    # projection coefficient.  Smooth confidence avoids hard X seams.
    # The edge-locked field is already a signed least-squares fit to a globally
    # validated PWM template.  Treat no-harm as a veto, not a second strength
    # estimator: keep non-worsening proposals at nearly full authority and only
    # fade them when their local template projection clearly points the wrong way
    # or increases the residual.  This avoids needlessly suppressing a real PWM
    # correction on textured subjects where the coherent stripe is small relative
    # to scene detail.
    direction = _smoothstep(-signed, 0.0005, 0.010)
    gate_x = _smoothstep(improvement, -0.100, -0.010) * direction
    # Coverage has already been incorporated into the direct X-only fit
    # evidence.  Do not multiply it in again here or low-texture/partially
    # supported subjects get needlessly attenuated a second time.
    gate_x = torch.where((before > 1e-12) & (den > 1e-12), gate_x, torch.zeros_like(gate_x))
    return gate_x.clamp(0.0, 1.0).expand(-1, 1, residual.shape[-2], -1)


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


def _broad_neural_guided_consensus_equalizer(
    y: torch.Tensor,
    neural_gain_hint: torch.Tensor,
    *,
    edge_support: torch.Tensor,
    raw_tone_support: torch.Tensor,
    luma_strength: float = 1.0,
    region_count: int = 6,
    min_regions: int = 2,
    corr_low: float = 0.55,
    corr_high: float = 0.80,
    smooth_fraction: float = 0.015,
    row_sigma: float = 2.0,
    huber_k: float = 2.5,
    min_coverage: float = 0.04,
    eps: float = 0.02,
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    """Remove a sub-cycle broad residual shared across multiple scene regions.

    The cumulative neural luminance gain is used as a *directional validator*,
    not as the correction itself.  This matters for the hardest broad cases:
    the network often points at the correct dark/bright trough but underestimates
    its amplitude.  Several widely separated scene strips must independently
    retain a broad residual opposite to that neural gain before the stage is
    allowed to act.

    Each scene strip receives its own affine illumination baseline.  The
    surviving broad residuals are robustly combined into one row-coherent
    waveform, and a bounded least-squares scale removes only the shared part.
    Unlike the first consensus implementation, no quadratic baseline or
    very-slow high-pass is used, because those operations erase the exact
    less-than-one-cycle dark patch this mode exists to recover.
    """
    if luma_strength <= 0:
        z = torch.zeros_like(y)
        return y, 0.0, z, z, z, 0.0, 0
    if y.ndim != 4 or y.shape[1] != 1:
        raise ValueError("Expected Y Bx1xHxW")
    if neural_gain_hint.shape != y.shape:
        raise ValueError("neural_gain_hint must match Y")

    b, _, h, w = y.shape
    logy = torch.log(y.float().clamp_min(0.0) + float(eps))
    guide_map = neural_gain_hint.float()
    base_support = (edge_support * raw_tone_support).clamp(0.0, 1.0)
    smooth_sigma = max(1.0, float(h) * float(smooth_fraction))

    corr_rows = torch.zeros((b, 1, h, 1), device=y.device, dtype=torch.float32)
    evidence_rows = torch.zeros_like(corr_rows)
    confidence_map = torch.zeros_like(y, dtype=torch.float32)
    accepted_total = 0
    conf_total = 0.0

    edges = [int(round(i * w / float(region_count))) for i in range(region_count + 1)]
    min_sep = max(1, region_count // 2)

    for bi in range(b):
        grow, gvalid, gconf, _ = _robust_row_consensus(
            guide_map[bi:bi+1], base_support[bi:bi+1],
            huber_k=huber_k, min_coverage=min_coverage,
        )
        gweight = (gvalid * gconf).clamp(0.0, 1.0)
        if float(gweight.mean()) < 0.08:
            continue
        if row_sigma > 0:
            grow = _normalized_smooth_axis(grow, gvalid, float(row_sigma), "y")
        gbase = _weighted_poly_baseline(grow, gweight, degree=1)
        guide = _smooth_axis(grow - gbase, smooth_sigma, "y")
        guide = guide - guide.mean(dim=-2, keepdim=True)
        guide_1d = guide[0, 0, :, 0]
        if not math.isfinite(float(guide_1d.std())) or float(guide_1d.std()) < 1e-4:
            continue

        residuals: list[torch.Tensor] = []
        row_weights: list[torch.Tensor] = []
        scalar_weights: list[float] = []
        region_indices: list[int] = []
        anti_scores: list[float] = []

        for ri in range(region_count):
            x0, x1 = edges[ri], edges[ri + 1]
            if x1 - x0 < 8:
                continue
            yy = logy[bi:bi+1, :, :, x0:x1]
            ss = base_support[bi:bi+1, :, :, x0:x1]
            if float(ss.mean()) < max(0.01, float(min_coverage) * 0.35):
                continue
            row, valid, conf, _ = _robust_row_consensus(
                yy, ss, huber_k=huber_k, min_coverage=min_coverage
            )
            fit_weight = (valid * conf).clamp(0.0, 1.0)
            if float(fit_weight.mean()) < 0.08:
                continue
            if row_sigma > 0:
                row = _normalized_smooth_axis(row, valid, float(row_sigma), "y")

            # Affine-only baseline: preserve a single broad trough/hump even if
            # less than one full cycle is visible in the frame.
            slow = _weighted_poly_baseline(row, fit_weight, degree=1)
            residual = _smooth_axis(row - slow, smooth_sigma, "y")
            residual = residual - residual.mean(dim=-2, keepdim=True)
            r = residual[0, 0, :, 0]
            rw = fit_weight[0, 0, :, 0] * gweight[0, 0, :, 0]
            if float(rw.mean()) < 0.05 or float(r.std()) < 2e-4:
                continue

            wsum = rw.sum().clamp_min(1e-6)
            ra = r - (r * rw).sum() / wsum
            ga = guide_1d - (guide_1d * rw).sum() / wsum
            denom = torch.sqrt((ra.square() * rw).sum() * (ga.square() * rw).sum()).clamp_min(1e-8)
            anti = float((-(ra * ga * rw).sum() / denom).clamp(-1.0, 1.0))
            if anti < corr_low:
                continue

            t = max(0.0, min(1.0, (anti - corr_low) / (corr_high - corr_low)))
            corr_gate = t * t * (3.0 - 2.0 * t)
            residuals.append(r)
            row_weights.append(rw)
            scalar_weights.append(max(1e-4, float(rw.mean())) * max(0.20, corr_gate))
            region_indices.append(ri)
            anti_scores.append(anti)

        if len(residuals) < min_regions:
            continue
        if max(region_indices) - min(region_indices) < min_sep:
            continue

        stack = torch.stack(residuals, dim=0)
        med = stack.median(dim=0).values
        mad = (stack - med).abs().median(dim=0).values.clamp_min(2e-4)
        clipped = torch.maximum(torch.minimum(stack, med + 3.0 * mad), med - 3.0 * mad)
        ww = torch.tensor(scalar_weights, device=y.device, dtype=torch.float32).view(-1, 1)
        consensus = (clipped * ww).sum(dim=0) / ww.sum().clamp_min(1e-6)
        consensus = _smooth_axis(consensus.view(1, 1, h, 1), smooth_sigma, "y")[0, 0, :, 0]
        consensus = consensus - consensus.mean()
        if float(consensus.std()) < 2e-4:
            continue

        # Final guide check on the actual proposed waveform.
        gw = gweight[0, 0, :, 0]
        gws = gw.sum().clamp_min(1e-6)
        ca = consensus - (consensus * gw).sum() / gws
        ga = guide_1d - (guide_1d * gw).sum() / gws
        gden = torch.sqrt((ca.square() * gw).sum() * (ga.square() * gw).sum()).clamp_min(1e-8)
        guide_anti = float((-(ca * ga * gw).sum() / gden).clamp(-1.0, 1.0))
        if guide_anti < corr_low:
            continue

        # No-harm amplitude: fit how much of the common residual should be
        # cancelled across the accepted regions.  User strength is a maximum
        # authority, not permission to overshoot the fitted minimum-energy
        # solution. This makes strength 2 safe when the optimum is near 1.
        numer = torch.zeros((), device=y.device, dtype=torch.float32)
        denom = torch.zeros_like(numer)
        for r, rw in zip(residuals, row_weights):
            numer = numer + (r * consensus * rw).sum()
            denom = denom + (consensus.square() * rw).sum()
        raw_fit = float((numer / denom.clamp_min(1e-8)).clamp(0.0, 2.0))
        scale = min(raw_fit, float(luma_strength))
        if scale <= 1e-4:
            continue

        correction = (-consensus * scale).view(1, 1, h, 1)
        corr_rows[bi:bi+1] = correction
        evidence_rows[bi:bi+1] = consensus.abs().view(1, 1, h, 1)
        mean_score = (sum(anti_scores) + guide_anti) / (len(anti_scores) + 1)
        tt = max(0.0, min(1.0, (mean_score - corr_low) / (corr_high - corr_low)))
        corr_conf = tt * tt * (3.0 - 2.0 * tt)
        region_conf = min(1.0, len(residuals) / float(region_count))
        global_conf = max(corr_conf, region_conf)
        confidence_map[bi:bi+1].fill_(global_conf)
        accepted_total += len(residuals)
        conf_total += global_conf

    corr_map = corr_rows.expand(-1, -1, -1, w)
    # Positive broad recovery should not blow near-white windows/speculars. The
    # correction remains fully row-coherent through shadows and midtones.
    bright_guard = 1.0 - _smoothstep(y.float(), 0.88, 0.985)
    applied_corr = torch.where(corr_map > 0.0, corr_map * bright_guard, corr_map)
    out_y = (y.float() + float(eps)) * torch.exp(applied_corr) - float(eps)

    # A validated first consensus can leave a small under-corrected remainder
    # after its bounded no-harm fit. Re-measure that residual once, using the
    # already validated first-pass waveform as the template. Because the first
    # pass already proved cross-surface agreement, the refinement only needs two
    # supporting regions; it must still match the same waveform and pass a new
    # least-squares no-harm fit. Extra authority is deliberately small and is
    # never exposed as another GUI knob.
    refine_limit = min(0.35, max(0.0, float(luma_strength)) * 0.35)
    refine_corr_rows = torch.zeros_like(corr_rows)
    refine_evidence_rows = torch.zeros_like(evidence_rows)
    if accepted_total >= min_regions and refine_limit > 1e-4:
        (
            out_y, refine_corr_rows, refine_evidence_rows, _refine_conf, _refine_regions
        ) = _broad_luma_template_refinement(
            out_y, corr_rows,
            edge_support=edge_support, raw_tone_support=raw_tone_support,
            max_strength=refine_limit, region_count=region_count,
            min_regions=min_regions,
            corr_low=max(0.25, float(corr_low) - 0.25),
            corr_high=max(max(0.25, float(corr_low) - 0.25) + 0.15, float(corr_high) - 0.20),
            smooth_fraction=smooth_fraction, row_sigma=row_sigma,
            huber_k=huber_k, min_coverage=min_coverage, eps=eps,
        )

    total_corr_rows = corr_rows + refine_corr_rows
    total_log_delta = torch.log(
        (out_y.float() + float(eps)).clamp_min(1e-6)
        / (y.float() + float(eps)).clamp_min(1e-6)
    )
    yrms = float(total_log_delta.square().mean().sqrt())
    batches_with_consensus = max(1, int((confidence_map[:, :, :1, :1] > 0).sum()))
    confidence = conf_total / batches_with_consensus if accepted_total else 0.0
    apply = confidence_map
    evidence = torch.maximum(evidence_rows, refine_evidence_rows).expand(-1, -1, -1, w)
    return out_y, yrms, total_corr_rows, apply, evidence, float(confidence), int(accepted_total)


def _broad_luma_template_refinement(
    y: torch.Tensor,
    correction_template: torch.Tensor,
    *,
    edge_support: torch.Tensor,
    raw_tone_support: torch.Tensor,
    max_strength: float = 0.35,
    region_count: int = 6,
    min_regions: int = 2,
    corr_low: float = 0.30,
    corr_high: float = 0.60,
    smooth_fraction: float = 0.015,
    row_sigma: float = 2.0,
    huber_k: float = 2.5,
    min_coverage: float = 0.04,
    eps: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    """Apply one low-authority residual refinement after broad Y consensus.

    ``correction_template`` is the already validated first-pass row correction.
    The residual template is therefore its opposite sign. Each current scene
    strip gets a fresh affine baseline; only strips whose remaining broad
    residual still matches that template are retained. The combined residual
    must itself match the same waveform, and a second bounded least-squares fit
    limits the extra correction. This is intentionally a refinement, not an
    independent detector, so it cannot activate when the first consensus did
    not already establish the broad flicker waveform.
    """
    if max_strength <= 0:
        z = torch.zeros((y.shape[0], 1, y.shape[-2], 1), device=y.device, dtype=torch.float32)
        return y, z, z, 0.0, 0
    if y.ndim != 4 or y.shape[1] != 1:
        raise ValueError("Expected Y Bx1xHxW")
    if correction_template.ndim != 4 or correction_template.shape[0] != y.shape[0] or correction_template.shape[1] != 1 or correction_template.shape[-2] != y.shape[-2]:
        raise ValueError("correction_template must be Bx1xHx1 or Bx1xHxW")

    b, _, h, w = y.shape
    logy = torch.log(y.float().clamp_min(0.0) + float(eps))
    base_support = (edge_support * raw_tone_support).clamp(0.0, 1.0)
    smooth_sigma = max(1.0, float(h) * float(smooth_fraction))
    corr_rows = torch.zeros((b, 1, h, 1), device=y.device, dtype=torch.float32)
    evidence_rows = torch.zeros_like(corr_rows)
    confidence_total = 0.0
    accepted_total = 0
    edges = [int(round(i * w / float(region_count))) for i in range(region_count + 1)]

    for bi in range(b):
        template_corr = correction_template[bi:bi+1].float()
        if template_corr.shape[-1] != 1:
            template_corr = template_corr.mean(dim=-1, keepdim=True)
        # What remains after an under-corrected first pass has the opposite sign
        # of the applied correction template. Amplitude is irrelevant here; the
        # template is used only for waveform/direction validation.
        template = -template_corr[0, 0, :, 0]
        template = template - template.mean()
        if not math.isfinite(float(template.std())) or float(template.std()) < 2e-4:
            continue

        residuals: list[torch.Tensor] = []
        row_weights: list[torch.Tensor] = []
        scalar_weights: list[float] = []
        scores: list[float] = []

        for ri in range(region_count):
            x0, x1 = edges[ri], edges[ri + 1]
            if x1 - x0 < 8:
                continue
            yy = logy[bi:bi+1, :, :, x0:x1]
            ss = base_support[bi:bi+1, :, :, x0:x1]
            if float(ss.mean()) < max(0.01, float(min_coverage) * 0.35):
                continue
            row, valid, conf, _ = _robust_row_consensus(
                yy, ss, huber_k=huber_k, min_coverage=min_coverage
            )
            fit_weight = (valid * conf).clamp(0.0, 1.0)
            if float(fit_weight.mean()) < 0.08:
                continue
            if row_sigma > 0:
                row = _normalized_smooth_axis(row, valid, float(row_sigma), "y")
            slow = _weighted_poly_baseline(row, fit_weight, degree=1)
            residual = _smooth_axis(row - slow, smooth_sigma, "y")
            residual = residual - residual.mean(dim=-2, keepdim=True)
            r = residual[0, 0, :, 0]
            rw = fit_weight[0, 0, :, 0]
            if float(rw.mean()) < 0.05 or float(r.std()) < 1e-4:
                continue

            wsum = rw.sum().clamp_min(1e-6)
            ra = r - (r * rw).sum() / wsum
            ta = template - (template * rw).sum() / wsum
            denom = torch.sqrt((ra.square() * rw).sum() * (ta.square() * rw).sum()).clamp_min(1e-8)
            score = float(((ra * ta * rw).sum() / denom).clamp(-1.0, 1.0))
            if score < corr_low:
                continue
            t = max(0.0, min(1.0, (score - corr_low) / (corr_high - corr_low)))
            gate = t * t * (3.0 - 2.0 * t)
            residuals.append(r)
            row_weights.append(rw)
            scalar_weights.append(max(1e-4, float(rw.mean())) * max(0.20, gate))
            scores.append(score)

        # The first pass already proved that the waveform is shared across
        # widely separated surfaces. Requiring that separation again here can
        # suppress exactly the local remainder that a refinement should finish.
        if len(residuals) < min_regions:
            continue

        stack = torch.stack(residuals, dim=0)
        med = stack.median(dim=0).values
        mad = (stack - med).abs().median(dim=0).values.clamp_min(1e-4)
        clipped = torch.maximum(torch.minimum(stack, med + 3.0 * mad), med - 3.0 * mad)
        ww = torch.tensor(scalar_weights, device=y.device, dtype=torch.float32).view(-1, 1)
        consensus = (clipped * ww).sum(dim=0) / ww.sum().clamp_min(1e-6)
        consensus = _smooth_axis(consensus.view(1, 1, h, 1), smooth_sigma, "y")[0, 0, :, 0]
        consensus = consensus - consensus.mean()
        if float(consensus.std()) < 1e-4:
            continue

        # Strong same-waveform check against the validated first-pass residual.
        den = torch.sqrt(consensus.square().sum() * template.square().sum()).clamp_min(1e-8)
        template_corr_score = float(((consensus * template).sum() / den).clamp(-1.0, 1.0))
        if template_corr_score < max(0.40, corr_low + 0.10):
            continue

        # New no-harm fit on the remaining residual. Any scale in [0, raw_fit]
        # moves toward the minimum-energy solution; the hard refinement cap keeps
        # this second pass subordinate to the primary consensus.
        numer = torch.zeros((), device=y.device, dtype=torch.float32)
        denom = torch.zeros_like(numer)
        for r, rw in zip(residuals, row_weights):
            numer = numer + (r * consensus * rw).sum()
            denom = denom + (consensus.square() * rw).sum()
        raw_fit = float((numer / denom.clamp_min(1e-8)).clamp(0.0, 2.0))
        scale = min(raw_fit, float(max_strength))
        if scale <= 1e-4:
            continue

        correction = (-consensus * scale).view(1, 1, h, 1)
        corr_rows[bi:bi+1] = correction
        evidence_rows[bi:bi+1] = consensus.abs().view(1, 1, h, 1)
        mean_score = (sum(scores) + template_corr_score) / (len(scores) + 1)
        tt = max(0.0, min(1.0, (mean_score - corr_low) / (corr_high - corr_low)))
        confidence_total += tt * tt * (3.0 - 2.0 * tt)
        accepted_total += len(residuals)

    if accepted_total == 0:
        return y, corr_rows, evidence_rows, 0.0, 0

    corr_map = corr_rows.expand(-1, -1, -1, w)
    bright_guard = 1.0 - _smoothstep(y.float(), 0.88, 0.985)
    applied_corr = torch.where(corr_map > 0.0, corr_map * bright_guard, corr_map)
    out_y = (y.float() + float(eps)) * torch.exp(applied_corr) - float(eps)
    confidence = confidence_total / max(1, b)
    return out_y, corr_rows, evidence_rows, float(confidence), int(accepted_total)


def _broad_neural_guided_chroma_consensus_equalizer(
    c: torch.Tensor,
    neural_chroma_hint: torch.Tensor,
    *,
    edge_support: torch.Tensor,
    raw_tone_support: torch.Tensor,
    chroma_strength: float = 1.0,
    region_count: int = 6,
    min_regions: int = 2,
    corr_low: float = 0.55,
    corr_high: float = 0.80,
    smooth_fraction: float = 0.015,
    row_sigma: float = 2.0,
    huber_k: float = 2.5,
    min_coverage: float = 0.04,
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    """Remove a broad Cb/Cr residual shared across unrelated scene regions.

    This is the chroma companion to the neural-guided broad luminance stage.
    Cb and Cr are treated as one 2-D chroma vector: agreement is measured from
    the weighted vector dot product across both channels and rows, so a region
    whose hue-shift direction disagrees with the other regions cannot satisfy
    the consensus test merely because one channel happens to correlate.

    The cumulative source -> post-neural chroma delta is used only as an
    *association validator*.  Each scene strip gets an independent affine Cb/Cr
    baseline, the remaining broad vector residuals are robustly combined, and a
    bounded least-squares fit decides how much of that common vector waveform
    can be removed.  The user-selected chroma strength is therefore a maximum
    authority rather than a forced multiplier.

    The final correction is row-coherent and constant across processing X.  It
    deliberately avoids saturation/object-shaped application masks; those would
    turn a physically global illumination-color correction into a scene-shaped
    tint.  The authored GUI Broad mask still gates the final visible delta.
    """
    if chroma_strength <= 0:
        z2 = torch.zeros_like(c)
        z1 = torch.zeros_like(c[:, :1])
        return c, 0.0, z2, z1, z1, 0.0, 0
    if c.ndim != 4 or c.shape[1] != 2:
        raise ValueError("Expected CbCr Bx2xHxW")
    if neural_chroma_hint.shape != c.shape:
        raise ValueError("neural_chroma_hint must match CbCr")

    b, _, h, w = c.shape
    guide_map = neural_chroma_hint.float()
    base_support = (edge_support * raw_tone_support).clamp(0.0, 1.0)
    smooth_sigma = max(1.0, float(h) * float(smooth_fraction))

    corr_rows = torch.zeros((b, 2, h, 1), device=c.device, dtype=torch.float32)
    evidence_rows = torch.zeros((b, 1, h, 1), device=c.device, dtype=torch.float32)
    confidence_map = torch.zeros_like(c[:, :1], dtype=torch.float32)
    accepted_total = 0
    conf_total = 0.0

    edges = [int(round(i * w / float(region_count))) for i in range(region_count + 1)]
    min_sep = max(1, region_count // 2)

    def _weighted_vector_correlation(
        residual: torch.Tensor,
        guide: torch.Tensor,
        weight: torch.Tensor,
    ) -> float:
        """Return weighted cosine correlation of two row-varying chroma vectors."""
        wsum = weight.sum().clamp_min(1e-6)
        ra = residual - (residual * weight.unsqueeze(0)).sum(dim=1, keepdim=True) / wsum
        ga = guide - (guide * weight.unsqueeze(0)).sum(dim=1, keepdim=True) / wsum
        ww = weight.unsqueeze(0)
        dot = (ra * ga * ww).sum()
        den = torch.sqrt((ra.square() * ww).sum() * (ga.square() * ww).sum()).clamp_min(1e-10)
        return float((dot / den).clamp(-1.0, 1.0))

    for bi in range(b):
        grow, gvalid, gconf, _ = _robust_row_consensus(
            guide_map[bi:bi+1], base_support[bi:bi+1],
            huber_k=huber_k, min_coverage=min_coverage,
        )
        gweight = (gvalid * gconf).clamp(0.0, 1.0)
        if float(gweight.mean()) < 0.08:
            continue
        if row_sigma > 0:
            grow = _normalized_smooth_axis(grow, gvalid, float(row_sigma), "y")
        gbase = _weighted_poly_baseline(grow, gweight, degree=1)
        guide = _smooth_axis(grow - gbase, smooth_sigma, "y")
        guide = guide - guide.mean(dim=-2, keepdim=True)
        guide_2d = guide[0, :, :, 0]
        guide_rms = float(guide_2d.square().mean().sqrt())
        if not math.isfinite(guide_rms) or guide_rms < 2e-5:
            continue

        residuals: list[torch.Tensor] = []
        row_weights: list[torch.Tensor] = []
        scalar_weights: list[float] = []
        region_indices: list[int] = []
        guide_scores: list[float] = []

        for ri in range(region_count):
            x0, x1 = edges[ri], edges[ri + 1]
            if x1 - x0 < 8:
                continue
            cc = c[bi:bi+1, :, :, x0:x1].float()
            ss = base_support[bi:bi+1, :, :, x0:x1]
            if float(ss.mean()) < max(0.01, float(min_coverage) * 0.35):
                continue
            row, valid, conf, _ = _robust_row_consensus(
                cc, ss, huber_k=huber_k, min_coverage=min_coverage
            )
            fit_weight = (valid * conf).clamp(0.0, 1.0)
            if float(fit_weight.mean()) < 0.08:
                continue
            if row_sigma > 0:
                row = _normalized_smooth_axis(row, valid, float(row_sigma), "y")

            # Preserve a sub-cycle Cb/Cr cast by removing only an affine
            # per-region baseline, exactly as in the revised Y consensus path.
            slow = _weighted_poly_baseline(row, fit_weight, degree=1)
            residual = _smooth_axis(row - slow, smooth_sigma, "y")
            residual = residual - residual.mean(dim=-2, keepdim=True)
            r = residual[0, :, :, 0]
            rw = fit_weight[0, 0, :, 0] * gweight[0, 0, :, 0]
            if float(rw.mean()) < 0.05 or float(r.square().mean().sqrt()) < 2e-5:
                continue

            # Chroma can fail in either direction: an under-corrected cast is
            # opposite to the neural delta, while an overshot/model-introduced
            # cast can point in the same direction.  Require a strong vector
            # relationship to what the neural branch changed, but do not force
            # the sign. The actual correction still removes the measured
            # cross-surface residual, never the guide itself.
            signed_corr = _weighted_vector_correlation(r, guide_2d, rw)
            guide_score = abs(signed_corr)
            if guide_score < corr_low:
                continue
            t = max(0.0, min(1.0, (guide_score - corr_low) / (corr_high - corr_low)))
            corr_gate = t * t * (3.0 - 2.0 * t)
            residuals.append(r)
            row_weights.append(rw)
            scalar_weights.append(max(1e-4, float(rw.mean())) * max(0.20, corr_gate))
            region_indices.append(ri)
            guide_scores.append(guide_score)

        if len(residuals) < min_regions:
            continue

        # Because chroma accepts either under-correction (residual opposite the
        # neural delta) or over-correction/model-introduced cast (residual in the
        # same direction), an absolute guide correlation alone is not enough:
        # two regions with opposite color waveforms could otherwise both pass.
        # Require the retained residuals themselves to agree positively with a
        # widely separated partner before they are allowed into the consensus.
        candidate_stack = torch.stack(residuals, dim=0)  # Nx2xH
        centered = candidate_stack - candidate_stack.mean(dim=2, keepdim=True)
        flat = centered.reshape(centered.shape[0], -1)
        norms = torch.sqrt(flat.square().sum(dim=1).clamp_min(1e-12))
        pair_corr = (flat @ flat.T) / (norms[:, None] * norms[None, :]).clamp_min(1e-12)
        pair_corr.fill_diagonal_(-1.0)
        partner_need = min_regions - 1
        keep: list[int] = []
        pair_gates: list[float] = []
        for i in range(len(residuals)):
            far = [
                j for j in range(len(residuals))
                if j != i and abs(region_indices[i] - region_indices[j]) >= min_sep
            ]
            if len(far) < partner_need:
                continue
            far_idx = torch.tensor(far, device=c.device, dtype=torch.long)
            top = torch.topk(pair_corr[i].index_select(0, far_idx), k=partner_need).values
            if float(top.min()) < corr_low:
                continue
            pair_score = float(top.mean())
            tpair = max(0.0, min(1.0, (pair_score - corr_low) / (corr_high - corr_low)))
            keep.append(i)
            pair_gates.append(tpair * tpair * (3.0 - 2.0 * tpair))
        if len(keep) < min_regions:
            continue

        residuals = [residuals[i] for i in keep]
        row_weights = [row_weights[i] for i in keep]
        region_indices = [region_indices[i] for i in keep]
        guide_scores = [guide_scores[i] for i in keep]
        scalar_weights = [scalar_weights[i] * max(1e-3, pair_gates[k]) for k, i in enumerate(keep)]

        stack = torch.stack(residuals, dim=0)  # Nx2xH
        med = stack.median(dim=0).values
        mad = (stack - med).abs().median(dim=0).values.clamp_min(2e-5)
        clipped = torch.maximum(torch.minimum(stack, med + 3.0 * mad), med - 3.0 * mad)
        ww = torch.tensor(scalar_weights, device=c.device, dtype=torch.float32).view(-1, 1, 1)
        consensus = (clipped * ww).sum(dim=0) / ww.sum().clamp_min(1e-6)
        consensus = _smooth_axis(consensus.view(1, 2, h, 1), smooth_sigma, "y")[0, :, :, 0]
        # Explicitly preserve the global Cb/Cr DC level.
        consensus = consensus - consensus.mean(dim=1, keepdim=True)
        if float(consensus.square().mean().sqrt()) < 2e-5:
            continue

        gw = gweight[0, 0, :, 0]
        guide_corr = abs(_weighted_vector_correlation(consensus, guide_2d, gw))
        if guide_corr < corr_low:
            continue

        # One scalar fit preserves the common Cb/Cr vector direction instead of
        # fitting independent per-channel gains that could rotate the hue.
        numer = torch.zeros((), device=c.device, dtype=torch.float32)
        denom = torch.zeros_like(numer)
        for r, rw in zip(residuals, row_weights):
            ww_row = rw.unsqueeze(0)
            numer = numer + (r * consensus * ww_row).sum()
            denom = denom + (consensus.square() * ww_row).sum()
        raw_fit = float((numer / denom.clamp_min(1e-10)).clamp(0.0, 2.0))
        scale = min(raw_fit, float(chroma_strength))
        if scale <= 1e-4:
            continue

        correction = (-consensus * scale).view(1, 2, h, 1)
        corr_rows[bi:bi+1] = correction
        evidence_rows[bi:bi+1] = consensus.square().sum(dim=0).sqrt().view(1, 1, h, 1)
        mean_score = (sum(guide_scores) + guide_corr) / (len(guide_scores) + 1)
        tt = max(0.0, min(1.0, (mean_score - corr_low) / (corr_high - corr_low)))
        corr_conf = tt * tt * (3.0 - 2.0 * tt)
        region_conf = min(1.0, len(residuals) / float(region_count))
        global_conf = max(corr_conf, region_conf)
        confidence_map[bi:bi+1].fill_(global_conf)
        accepted_total += len(residuals)
        conf_total += global_conf

    corr_map = corr_rows.expand(-1, -1, -1, w)
    out_c = c.float() + corr_map
    crms = float(corr_map.square().mean().sqrt())
    batches_with_consensus = max(1, int((confidence_map[:, :, :1, :1] > 0).sum()))
    confidence = conf_total / batches_with_consensus if accepted_total else 0.0
    evidence = evidence_rows.expand(-1, -1, -1, w)
    return out_c, crms, corr_rows, confidence_map, evidence, float(confidence), int(accepted_total)


def _broad_consensus_row_equalizer(
    y: torch.Tensor,
    *,
    neural_gain_hint: torch.Tensor | None = None,
    edge_support: torch.Tensor,
    raw_tone_support: torch.Tensor,
    luma_strength: float = 1.0,
    region_count: int = 6,
    min_regions: int = 2,
    corr_low: float = 0.55,
    corr_high: float = 0.80,
    smooth_fraction: float = 0.015,
    baseline_fraction: float = 0.20,
    poly_degree: int = 2,
    row_sigma: float = 2.0,
    huber_k: float = 2.5,
    min_coverage: float = 0.04,
    eps: float = 0.02,
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
    """Remove a broad row-coherent luminance residual shared by unrelated surfaces.

    The processing image is divided into large regions across processing X.
    Each region receives its own robust row measurement and slow illumination
    baseline.  Only the broad residual waveform that agrees in phase across at
    least ``min_regions`` independent regions is retained.  The final correction
    is one-dimensional (constant across processing X), so automatic scene
    segmentation can influence *evidence* without drawing object-shaped
    corrections into the image.

    The fallback/legacy implementation below is luminance-only.  The normal GUI
    and CLI consensus path supplies neural Y/CbCr direction hints and therefore
    uses the newer guided luminance and chroma estimators.
    """
    if neural_gain_hint is not None:
        return _broad_neural_guided_consensus_equalizer(
            y, neural_gain_hint,
            edge_support=edge_support, raw_tone_support=raw_tone_support,
            luma_strength=luma_strength, region_count=region_count,
            min_regions=min_regions, corr_low=corr_low, corr_high=corr_high,
            smooth_fraction=smooth_fraction, row_sigma=row_sigma,
            huber_k=huber_k, min_coverage=min_coverage, eps=eps,
        )
    if luma_strength <= 0:
        z = torch.zeros_like(y)
        return y, 0.0, z, z, z, 0.0, 0
    if y.ndim != 4 or y.shape[1] != 1:
        raise ValueError("Expected Y Bx1xHxW")
    if region_count < 2:
        raise ValueError("broad consensus requires at least 2 regions")
    if min_regions < 2 or min_regions > region_count:
        raise ValueError("broad consensus min_regions must satisfy 2 <= min_regions <= region_count")
    if not (0.0 <= corr_low < corr_high <= 1.0):
        raise ValueError("broad consensus correlations must satisfy 0 <= low < high <= 1")
    if smooth_fraction <= 0 or baseline_fraction <= smooth_fraction:
        raise ValueError("broad consensus fractions must satisfy 0 < smooth < baseline")

    b, _, h, w = y.shape
    logy = torch.log(y.float().clamp_min(0.0) + float(eps))
    base_support = (edge_support * raw_tone_support).clamp(0.0, 1.0)
    smooth_sigma = max(1.0, float(h) * float(smooth_fraction))
    baseline_sigma = max(smooth_sigma * 2.0, float(h) * float(baseline_fraction))

    corr_rows = torch.zeros((b, 1, h, 1), device=y.device, dtype=torch.float32)
    evidence_rows = torch.zeros_like(corr_rows)
    confidence_map = torch.zeros_like(y, dtype=torch.float32)
    accepted_total = 0
    conf_total = 0.0

    # Fixed, non-overlapping large regions are intentional: they make the
    # evidence substantially more independent than overlapping scene-shaped
    # masks while still allowing wall/floor/ceiling bands to vote together.
    edges = [int(round(i * w / float(region_count))) for i in range(region_count + 1)]
    for bi in range(b):
        profiles: list[torch.Tensor] = []
        weights: list[float] = []
        profile_regions: list[int] = []
        for ri in range(region_count):
            x0, x1 = edges[ri], edges[ri + 1]
            if x1 - x0 < 8:
                continue
            yy = logy[bi:bi+1, :, :, x0:x1]
            ss = base_support[bi:bi+1, :, :, x0:x1]
            if float(ss.mean()) < max(0.01, float(min_coverage) * 0.35):
                continue
            row, valid, conf, _ = _robust_row_consensus(
                yy, ss, huber_k=huber_k, min_coverage=min_coverage
            )
            fit_weight = (valid * conf).clamp(0.0, 1.0)
            if float(fit_weight.mean()) < 0.08:
                continue
            if row_sigma > 0:
                row = _normalized_smooth_axis(row, valid, float(row_sigma), "y")
            slow_poly = _weighted_poly_baseline(row, fit_weight, degree=poly_degree)
            residual = row - slow_poly
            residual = _smooth_axis(residual, smooth_sigma, "y")
            # Remove only very slow composition/illumination drift.  The
            # difference retains the few-cycle band family without assuming a
            # sinusoid or a single FFT period.
            residual = residual - _smooth_axis(residual, baseline_sigma, "y")
            residual = residual - residual.mean(dim=-2, keepdim=True)
            sig = float(residual.std())
            if not math.isfinite(sig) or sig < 2e-4:
                continue
            profiles.append(residual[0, 0, :, 0])
            weights.append(max(1e-4, float(conf.mean()) * float(valid.mean())))
            profile_regions.append(ri)

        n = len(profiles)
        if n < min_regions:
            continue
        stack = torch.stack(profiles, dim=0).float()  # NxH
        centered = stack - stack.mean(dim=1, keepdim=True)
        norm = torch.sqrt((centered.square()).mean(dim=1).clamp_min(1e-12))
        corr = (centered @ centered.T) / float(h)
        corr = corr / (norm[:, None] * norm[None, :]).clamp_min(1e-12)
        corr.fill_diagonal_(-1.0)

        accepted: list[int] = []
        accept_weights: list[float] = []
        score_values: list[float] = []
        partner_need = min_regions - 1
        # Two adjacent tiles can still be pieces of the same wall. Require
        # consensus partners to be widely separated across processing X so a
        # single large surface cannot satisfy the multi-surface test by itself.
        min_sep = max(1, region_count // 2)
        for i in range(n):
            # profiles[] may omit weak regions, so preserve each profile's
            # original partition index for the separation test.
            far_js = [
                j for j in range(n)
                if j != i and abs(profile_regions[i] - profile_regions[j]) >= min_sep
            ]
            if len(far_js) < partner_need:
                continue
            far_idx = torch.tensor(far_js, device=corr.device, dtype=torch.long)
            vals = corr[i].index_select(0, far_idx)
            top = torch.topk(vals, k=partner_need).values
            if float(top.min()) < corr_low:
                continue
            score = float(top.mean())
            t = max(0.0, min(1.0, (score - corr_low) / (corr_high - corr_low)))
            gate = t * t * (3.0 - 2.0 * t)
            accepted.append(i)
            accept_weights.append(weights[i] * max(gate, 1e-3))
            score_values.append(score)

        if len(accepted) < min_regions:
            continue
        a = stack[accepted]
        # Robustly clip each contributing waveform around the cross-region
        # median before averaging. This rejects a region-specific broad object
        # gradient even when its overall correlation happened to pass.
        med = a.median(dim=0).values
        mad = (a - med).abs().median(dim=0).values.clamp_min(2e-4)
        clipped = torch.maximum(torch.minimum(a, med + 3.0 * mad), med - 3.0 * mad)
        ww = torch.tensor(accept_weights, device=y.device, dtype=torch.float32).view(-1, 1)
        consensus = (clipped * ww).sum(dim=0) / ww.sum().clamp_min(1e-6)
        consensus = consensus - consensus.mean()

        # Least-squares amplitude against the agreeing regions is a global
        # no-harm/authority estimate.  It may attenuate the waveform but never
        # silently amplify it beyond the user-selected strength.
        denom = (consensus.square().sum() * ww.sum()).clamp_min(1e-8)
        numer = ((a * consensus.view(1, -1)).sum(dim=1, keepdim=True) * ww).sum()
        amplitude = float((numer / denom).clamp(0.0, 1.0))
        mean_score = sum(score_values) / len(score_values)
        t = max(0.0, min(1.0, (mean_score - corr_low) / (corr_high - corr_low)))
        global_conf = t * t * (3.0 - 2.0 * t)
        # Partial consensus is deliberately conservative: confidence is squared
        # for correction authority, while high-confidence agreement remains
        # unchanged. This strongly attenuates coincidental broad scene trends.
        authority = global_conf * global_conf
        correction = (-consensus * amplitude * authority).view(1, 1, h, 1)
        corr_rows[bi:bi+1] = correction
        # Evidence is debug-only; it remains row-coherent by construction.
        evidence_rows[bi:bi+1] = consensus.abs().view(1, 1, h, 1)
        confidence_map[bi:bi+1].fill_(global_conf)
        accepted_total += len(accepted)
        conf_total += global_conf

    corr_map = corr_rows.expand(-1, -1, -1, w)
    # Apply the same log-gain across processing X.  Do not multiply by a 2-D
    # automatic scene mask here: the whole point is to avoid scene-shaped
    # broad corrections. The GUI's authored Broad mask still gates the final
    # delta outside this function.
    out_y = (y.float() + float(eps)) * torch.exp(corr_map * float(luma_strength)) - float(eps)
    yrms = float((corr_map * float(luma_strength)).square().mean().sqrt())
    batches_with_consensus = max(1, int((confidence_map[:, :, :1, :1] > 0).sum()))
    confidence = conf_total / batches_with_consensus if accepted_total else 0.0
    apply = confidence_map
    evidence = evidence_rows.expand(-1, -1, -1, w)
    return out_y, yrms, corr_rows, apply, evidence, float(confidence), int(accepted_total)


def _zero_weighted_mean(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.expand(-1, delta.shape[1], -1, -1)
    mean = (delta * w).sum(dim=(-2, -1), keepdim=True) / w.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return delta - mean


def _fill_row_profile(row: torch.Tensor, valid: torch.Tensor, sigma: float) -> torch.Tensor:
    return _normalized_smooth_axis(row, valid, max(1.0, sigma), "y")



def _circular_smooth_profile(profile: torch.Tensor, sigma_bins: float) -> torch.Tensor:
    """Small circular Gaussian smoother for CxN phase profiles."""
    if sigma_bins <= 0 or profile.shape[-1] < 3:
        return profile.float()
    sigma = float(max(0.25, sigma_bins))
    radius = max(1, int(math.ceil(3.0 * sigma)))
    radius = min(radius, max(1, profile.shape[-1] // 4))
    x = torch.arange(-radius, radius + 1, device=profile.device, dtype=profile.dtype)
    kernel = torch.exp(-0.5 * (x / sigma).square())
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    c = profile.shape[0]
    padded = F.pad(profile.unsqueeze(0), (radius, radius), mode="circular")
    return F.conv1d(padded, kernel.view(1, 1, -1).expand(c, 1, -1), groups=c).squeeze(0)




def _pwm_residual_edge_signal(
    y: torch.Tensor,
    *,
    max_h: int = 1024,
    max_w: int = 512,
) -> tuple[torch.Tensor, int] | None:
    """Build a scene-resistant signed transition signal from post-neural Y.

    PWM residuals are best recognized by their *repeated row-wide transitions*,
    not by ordinary image-spectrum power.  This helper lightly smooths texture,
    differentiates along Y, robustly normalizes each processing column by its
    own derivative MAD, then takes the cross-column median.  A real rolling PWM
    edge therefore reinforces across much of the frame while localized object
    boundaries remain minority outliers.

    Returns ``(signal, work_h)`` where signal is Bx1x(H-1)x1 in the resized
    analysis domain.  ``None`` means the residual does not contain enough
    coherent derivative structure to be useful as a timing cue.
    """
    if y.ndim != 4 or y.shape[1] != 1 or y.shape[-2] < 16 or y.shape[-1] < 8:
        return None
    h, w = y.shape[-2:]
    scale = min(1.0, float(max_h) / float(h), float(max_w) / float(w))
    if scale < 1.0:
        hh = max(32, int(round(h * scale)))
        ww = max(32, int(round(w * scale)))
        yy = F.interpolate(y.float(), size=(hh, ww), mode="area")
    else:
        yy = y.float()
        hh = h

    logy = torch.log(yy.clamp_min(0.0) + 0.02)
    # Suppress fine texture without broadening a PWM transition appreciably.
    logy = _smooth_axis(logy, 3.0, "x")
    logy = _smooth_axis(logy, 0.85, "y")
    d = logy[:, :, 1:, :] - logy[:, :, :-1, :]

    med = d.median(dim=-2, keepdim=True).values
    mad = (d - med).abs().median(dim=-2, keepdim=True).values
    scale_col = (1.4826 * mad).clamp_min(1.5e-3)
    z = ((d - med) / scale_col).clamp(-6.0, 6.0)

    # Only reject genuinely unusable darkness.  The gate is deliberately slow
    # and permissive so the PWM dark state does not remove its own edge evidence.
    slow = _smooth_axis(yy, max(3.0, 0.035 * float(hh)), "y")
    slow = _smooth_axis(slow, 2.0, "x")
    tone = _smoothstep(torch.minimum(slow[:, :, 1:, :], slow[:, :, :-1, :]), 0.010, 0.045)
    masked = torch.where(tone > 0.05, z, torch.full_like(z, float("nan")))
    row = torch.nanmedian(masked, dim=-1, keepdim=True).values
    row = torch.nan_to_num(row, nan=0.0)
    row = _smooth_axis(row, 0.85, "y")
    row = row - row.mean(dim=-2, keepdim=True)
    if float(row.square().mean().sqrt()) < 0.035:
        return None
    return row, int(hh)


def _pwm_residual_edge_period(
    y: torch.Tensor,
    *,
    full_height: int,
    min_period_px: float = 12.0,
    max_period_fraction: float = 0.60,
) -> BandPeriodEstimate | None:
    """Estimate PWM period from repeated row-wide residual transition pulses.

    This is the residual-side counterpart to ``_pwm_neural_edge_period``.  It is
    intentionally derivative/coherence based so broad scene illumination and
    low-frequency composition cannot win merely because a square wave spreads
    power over many Fourier harmonics.
    """
    built = _pwm_residual_edge_signal(y)
    if built is None:
        return None
    sig, work_h = built
    v = sig.reshape(-1)
    if v.numel() < 24:
        return None

    min_lag = max(3, int(round(float(min_period_px) * work_h / float(full_height))))
    max_full = min(float(full_height) * float(max_period_fraction), float(full_height) / 2.0)
    max_lag = min(work_h // 2, int(round(max_full * work_h / float(full_height))))
    if max_lag <= min_lag + 2:
        return None

    scores = torch.full((max_lag + 1,), -1.0, device=v.device, dtype=torch.float32)
    # P2: score one lag below min_lag so both the local-maximum test and the
    # parabolic sub-pixel refinement have a valid left neighbour at the floor.
    for lag in range(max(1, min_lag - 1), max_lag + 1):
        a = v[:-lag]
        b = v[lag:]
        den = torch.sqrt((a.square().sum() * b.square().sum()).clamp_min(1e-12))
        scores[lag] = (a * b).sum() / den

    local = torch.full_like(scores, -1.0)
    if max_lag - min_lag >= 2:
        # P2: min_lag itself must be selectable.  The previous window started at
        # min_lag+1, so a true period landing exactly on the search floor could
        # never win (jDOn4whY...: min_lag computes to 12, true lag is 12.10).
        lo_i = max(1, min_lag)
        mid = scores[lo_i:max_lag]
        keep = (mid >= scores[lo_i - 1:max_lag - 1]) & (mid >= scores[lo_i + 1:max_lag + 1])
        local[lo_i:max_lag] = torch.where(keep, mid, torch.full_like(mid, -1.0))
    best_lag = int(torch.argmax(local).item())
    best_score = float(local[best_lag])
    if best_score < 0.16:
        return None

    # v8 fundamental-first autocorrelation logic.  Dense straight PWM can create
    # an almost-flat ladder of excellent peaks at P, 2P, 3P ... and even very
    # large lags.  Picking the absolute maximum then selects an arbitrary member
    # of that plateau (the singer regression selected ~169/571 px instead of
    # ~21 px).  When several *local maxima* are essentially tied, prefer the
    # shortest one that still contains two opposite recurrent transition phases.
    # This is deliberately conservative: a smaller candidate must be within
    # 1.5% of the best autocorrelation score and have >=5 visible cycles.  Broad
    # PWM with only a few cycles therefore keeps its stronger broad peak.
    near_floor = max(0.16, 0.985 * best_score)
    near_lags = [
        lag for lag in range(min_lag + 1, max_lag)
        if float(local[lag]) >= near_floor
    ]
    for lag in near_lags[:32]:
        period_try = float(lag) * float(full_height) / float(work_h)
        if float(full_height) / max(period_try, 1e-6) < 5.0:
            continue
        phases = _pwm_residual_transition_phases(
            y, full_height=full_height, period=period_try,
            min_duty=0.08, max_duty=0.92,
        )
        if phases is None or float(phases[2]) < 0.10:
            continue
        best_lag = int(lag)
        best_score = float(local[lag])
        break

    refined = float(best_lag)
    # P2: allow sub-pixel refinement at the search floor now that scores[]
    # extends one lag lower.
    if max(1, min_lag - 1) < best_lag < max_lag:
        ym1, y0, yp1 = float(scores[best_lag - 1]), float(scores[best_lag]), float(scores[best_lag + 1])
        denom = ym1 - 2.0 * y0 + yp1
        if abs(denom) > 1e-8:
            refined += max(-0.5, min(0.5, 0.5 * (ym1 - yp1) / denom))

    period_full = refined * float(full_height) / float(work_h)
    if period_full < float(min_period_px) or period_full > max_full:
        return None
    confidence = max(0.0, min(1.0, (best_score - 0.14) / 0.45))

    candidates: list[tuple[float, float]] = []
    span = local[min_lag:max_lag + 1]
    top = torch.topk(span, k=min(12, span.numel())).indices + min_lag
    for lag_t in top:
        lag = int(lag_t.item())
        sc = max(0.0, float(local[lag]))
        candidates.append((lag * float(full_height) / float(work_h), sc))
    return BandPeriodEstimate(float(period_full), float(confidence), tuple(candidates))


def _pwm_residual_transition_phases(
    y: torch.Tensor,
    *,
    full_height: int,
    period: float,
    min_duty: float,
    max_duty: float,
) -> tuple[float, float, float] | None:
    """Estimate the two visible PWM transition phases from residual edge coherence."""
    built = _pwm_residual_edge_signal(y)
    if built is None or period <= 2.0:
        return None
    sig, work_h = built
    v = sig[0, 0, :, 0]
    period_work = float(period) * float(work_h) / float(full_height)
    if period_work < 3.0:
        return None

    bins = max(24, min(512, int(round(period_work * 3.0))))
    yy = torch.arange(v.numel(), device=v.device, dtype=torch.float32)
    phase_idx = torch.floor(torch.remainder(yy / period_work, 1.0) * bins).long().clamp(0, bins - 1)
    den = torch.zeros(bins, device=v.device, dtype=torch.float32)
    num = torch.zeros_like(den)
    den.scatter_add_(0, phase_idx, torch.ones_like(v))
    num.scatter_add_(0, phase_idx, v)
    folded = num / den.clamp_min(1.0)
    folded = _circular_smooth_profile(folded.view(1, -1), 1.0).view(-1)

    p_pos = int(torch.argmax(folded).item())
    p_neg = int(torch.argmin(folded).item())
    dist = (p_neg - p_pos) % bins
    frac = float(dist) / float(bins)
    if not (float(min_duty) <= frac <= float(max_duty)):
        # The complementary state interval is physically equivalent because the
        # fitted template coefficient is signed.  Reject only if *both* arcs are
        # outside the configured duty limits.
        comp = 1.0 - frac
        if not (float(min_duty) <= comp <= float(max_duty)):
            return None

    med = folded.median()
    mad = (folded - med).abs().median().clamp_min(1e-4)
    noise = 1.4826 * mad
    edge_score = float(torch.minimum(folded[p_pos].abs(), folded[p_neg].abs()) / noise)
    if edge_score < 1.20:
        return None

    # Verify that both signed transitions actually recur through the image, not
    # merely in the folded average.  Sample a small neighborhood around each
    # predicted edge and require broad cycle coverage.
    def recurrence(phase_bin: int, sign: float) -> float:
        phase_work = float(phase_bin) / float(bins) * period_work
        vals = []
        n0 = int(math.floor((0.0 - phase_work) / period_work)) - 1
        n1 = int(math.ceil((float(v.numel() - 1) - phase_work) / period_work)) + 1
        for n in range(n0, n1 + 1):
            pos = phase_work + n * period_work
            if pos < 2.0 or pos > float(v.numel() - 3):
                continue
            j = int(round(pos))
            sample = v[max(0, j - 2):min(v.numel(), j + 3)] * float(sign)
            vals.append(float(sample.max()))
        if len(vals) < 4:
            return 0.0
        vv = torch.tensor(vals, device=v.device)
        return float((vv > 0.18).float().mean())

    cov_pos = recurrence(p_pos, +1.0)
    cov_neg = recurrence(p_neg, -1.0)
    coverage = math.sqrt(max(0.0, cov_pos * cov_neg))
    if coverage < 0.35:
        return None
    confidence = max(0.0, min(1.0, (edge_score - 1.0) / 3.0)) * min(1.0, coverage / 0.75)
    return (float(p_pos) / float(bins), float(p_neg) / float(bins), float(confidence))


def _pwm_neural_edge_period(
    luma_field: torch.Tensor | None,
    chroma_delta_hint: torch.Tensor | None,
    *,
    full_height: int,
    min_period_px: float = 12.0,
    max_period_fraction: float = 0.60,
) -> BandPeriodEstimate | None:
    """Estimate a PWM period from the neural *correction* edge pulse train.

    Square-ish LED/PWM banding is rich in harmonics, so the ordinary spectrum
    can prefer a very broad scene trend or a harmonic of the true state period.
    The neural correction is a much cleaner physical hint: repeated LED state
    changes appear as same-polarity derivative pulses one full PWM period apart,
    while real scene edges are largely absent.  This detector therefore works
    on the row derivative of the neural Y/CbCr correction and uses normalized
    positive-lag autocorrelation.

    The returned period is in full-resolution processing pixels.  ``None`` means
    the edge train was too weak/ambiguous and the established period detector
    should remain authoritative.
    """
    if full_height < 24:
        return None

    signals: list[torch.Tensor] = []
    weights: list[float] = []
    analysis_h = min(1024, int(full_height))

    def add_hint(x: torch.Tensor | None, weight: float) -> None:
        if x is None or x.ndim != 4 or x.shape[-2] < 4 or x.shape[-1] < 1:
            return
        xx = x.detach().float()
        # Robust collapse across processing X.  Neural correction fields should
        # be row-coherent, but median collapse prevents a local model mistake or
        # bright fixture from defining the global PWM edge train.
        row = xx.median(dim=-1, keepdim=True).values
        row = _resize_row_signal(row, analysis_h)
        # Remove only affine drift; broad PWM plateaus themselves must survive.
        row = _remove_linear_y_trend(row, None)
        for ci in range(row.shape[1]):
            d = row[:, ci:ci+1, 1:, :] - row[:, ci:ci+1, :-1, :]
            med = d.median(dim=-2, keepdim=True).values
            mad = (d - med).abs().median(dim=-2, keepdim=True).values
            scale = (1.4826 * mad).clamp_min(1e-5)
            z = ((d - med) / scale).clamp(-8.0, 8.0)
            if float(z.square().mean().sqrt()) < 0.08:
                continue
            signals.append(z)
            weights.append(float(weight))

    add_hint(luma_field, 1.0)
    add_hint(chroma_delta_hint, 0.45)
    if not signals:
        return None

    min_lag = max(3, int(round(float(min_period_px) * analysis_h / float(full_height))))
    max_full = min(float(full_height) * float(max_period_fraction), float(full_height) / 2.0)
    max_lag = min(analysis_h // 2, int(round(max_full * analysis_h / float(full_height))))
    if max_lag <= min_lag + 2:
        return None

    scores = torch.zeros(max_lag + 1, device=signals[0].device, dtype=torch.float32)
    wsum = 0.0
    for sig, wt in zip(signals, weights):
        v = sig.reshape(-1)
        for lag in range(min_lag, max_lag + 1):
            a = v[:-lag]
            b = v[lag:]
            den = torch.sqrt((a.square().sum() * b.square().sum()).clamp_min(1e-12))
            scores[lag] += float(wt) * (a * b).sum() / den
        wsum += float(wt)
    scores /= max(wsum, 1e-6)
    scores[:min_lag] = -1.0

    # Prefer a genuine local autocorrelation maximum, not the monotonic shoulder
    # immediately beside zero lag.
    local = scores.clone()
    if max_lag - min_lag >= 2:
        # P2: same floor-exclusion fix as the residual detector.
        _lo = max(1, min_lag)
        local[_lo:max_lag] = torch.where(
            (scores[_lo:max_lag] >= scores[_lo - 1:max_lag - 1])
            & (scores[_lo:max_lag] >= scores[_lo + 1:max_lag + 1]),
            scores[_lo:max_lag],
            torch.full_like(scores[_lo:max_lag], -1.0),
        )
    best_lag = int(torch.argmax(local).item())
    best_score = float(local[best_lag])
    if best_score < 0.10:
        return None

    # If a smaller divisor is nearly as coherent, it is the actual PWM period;
    # the larger peak is simply a repeated-cycle harmonic (2P, 3P, ...).
    for div in (4, 3, 2):
        target = best_lag / float(div)
        lo = max(min_lag, int(math.floor(target)) - 1)
        hi = min(max_lag, int(math.ceil(target)) + 1)
        if hi < lo:
            continue
        cand = max(range(lo, hi + 1), key=lambda k: float(local[k]))
        if float(local[cand]) >= max(0.10, 0.68 * best_score):
            best_lag = int(cand)
            best_score = float(local[cand])

    # Sub-bin parabolic refinement around the selected autocorrelation peak.
    refined = float(best_lag)
    # P2: allow refinement at the search floor.
    if max(1, min_lag - 1) < best_lag < max_lag:
        ym1, y0, yp1 = (float(scores[best_lag - 1]), float(scores[best_lag]), float(scores[best_lag + 1]))
        denom = ym1 - 2.0 * y0 + yp1
        if abs(denom) > 1e-8:
            refined += max(-0.5, min(0.5, 0.5 * (ym1 - yp1) / denom))

    period_full = refined * float(full_height) / float(analysis_h)
    if period_full < float(min_period_px) or period_full > max_full:
        return None
    confidence = max(0.0, min(1.0, (best_score - 0.08) / 0.30))

    candidates: list[tuple[float, float]] = []
    top = torch.topk(local[min_lag:max_lag+1], k=min(12, max_lag-min_lag+1)).indices + min_lag
    for lag_t in top:
        lag = int(lag_t.item())
        sc = max(0.0, float(local[lag]))
        candidates.append((lag * float(full_height) / float(analysis_h), sc))
    return BandPeriodEstimate(float(period_full), float(confidence), tuple(candidates))



def _pwm_neural_transition_phases(
    luma_field: torch.Tensor | None,
    chroma_delta_hint: torch.Tensor | None,
    *,
    full_height: int,
    period: float,
    min_duty: float,
    max_duty: float,
) -> tuple[float, float, float] | None:
    """Return two opposite PWM edge phases from neural correction hints.

    The neural correction is used only to lock period/phase/duty.  Plateau
    amplitudes are still measured from the post-neural residual image, so this
    cannot simply re-apply the model prediction.
    """
    if period <= 2.0 or full_height < 12:
        return None
    bins = max(16, min(768, int(round(float(period)))))
    channels: list[torch.Tensor] = []

    def add(x: torch.Tensor | None) -> None:
        if x is None or x.ndim != 4 or x.shape[-2] < 4:
            return
        row = x.detach().float().median(dim=-1, keepdim=True).values
        row = _resize_row_signal(row, full_height)
        row = _remove_linear_y_trend(row, None)
        for ci in range(row.shape[1]):
            channels.append(row[:, ci:ci+1])

    # Luminance correction is the cleanest timing cue for LED state edges.
    # Chroma can be strongly surface-dependent, so use it for phase only when
    # no usable luminance hint is available.
    add(luma_field)
    if not channels:
        add(chroma_delta_hint)
    if not channels:
        return None

    # Limit to Y + first two chroma-like channels. Extra channels are irrelevant.
    rows = torch.cat(channels[:3], dim=1)
    h = rows.shape[-2]
    yy = torch.arange(h, device=rows.device, dtype=torch.float32)
    phase_idx = torch.floor(torch.remainder(yy / float(period), 1.0) * bins).long().clamp(0, bins - 1)
    den = torch.zeros(bins, device=rows.device, dtype=torch.float32)
    den.scatter_add_(0, phase_idx, torch.ones_like(yy))
    folded_channels: list[torch.Tensor] = []
    for ci in range(rows.shape[1]):
        vals = rows[0, ci, :, 0]
        phase_values = []
        for pi in range(bins):
            vv = vals[phase_idx == pi]
            phase_values.append(vv.median() if vv.numel() else torch.tensor(0.0, device=rows.device))
        folded_channels.append(torch.stack(phase_values))
    folded = torch.stack(folded_channels, dim=0)
    folded = _circular_smooth_profile(folded, 0.45)
    deriv = torch.roll(folded, shifts=-1, dims=-1) - folded

    med = deriv.median(dim=-1, keepdim=True).values
    mad = (deriv - med).abs().median(dim=-1, keepdim=True).values
    scale = (1.4826 * mad).clamp_min(1e-5)
    if scale.shape[0] > 1:
        floor = (0.12 * scale[0:1]).clamp_min(1e-4)
        scale[1:] = torch.maximum(scale[1:], floor.expand_as(scale[1:]))
    z = (deriv - med) / scale
    channel_weight = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
    if z.shape[0] > 1:
        channel_weight[1:] = 0.40
    grad = torch.sqrt((z.square() * channel_weight[:, None]).sum(dim=0))
    p1 = int(torch.argmax(grad).item())
    v1 = z[:, p1] * torch.sqrt(channel_weight)
    v1n = torch.sqrt(v1.square().sum()).clamp_min(1e-8)

    best: tuple[float, int, float] | None = None
    for p2 in range(bins):
        dist = (p2 - p1) % bins
        if dist < int(math.ceil(float(min_duty) * bins)) or dist > int(math.floor(float(max_duty) * bins)):
            continue
        v2 = z[:, p2] * torch.sqrt(channel_weight)
        v2n = torch.sqrt(v2.square().sum()).clamp_min(1e-8)
        opposition = float((-torch.dot(v1, v2) / (v1n * v2n)).clamp(0.0, 1.0))
        if opposition < 0.35:
            continue
        pair = float(torch.minimum(grad[p1], grad[p2])) * opposition
        if best is None or pair > best[0]:
            best = (pair, p2, opposition)
    if best is None:
        return None
    pair, p2, opposition = best

    gmed = float(grad.median())
    gmad = float((grad - grad.median()).abs().median()) * 1.4826
    noise = max(1e-6, gmed + gmad)
    score = pair / noise
    if score < 1.35:
        return None
    confidence = max(0.0, min(1.0, (score - 1.25) / 3.0)) * opposition
    return (float(p1) / bins, float(p2) / bins, float(confidence))





def _pwm_periods_harmonic_related(a: float, b: float, *, tol: float = 0.045) -> bool:
    """Return True when two periods belong to the same small-integer family.

    A square/PWM waveform naturally produces strong P/2, P/3... spectral or
    autocorrelation candidates.  Treating those as independent light sources
    would double-count the same band train.  Distinct non-harmonic periods are
    kept and may be fitted independently by the segmented multi-source model.
    """
    a = float(a)
    b = float(b)
    if a <= 0.0 or b <= 0.0:
        return False
    ratio = max(a, b) / max(1e-6, min(a, b))
    if abs(math.log(a / b)) < math.log(1.055):
        return True
    for k in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
        if abs(ratio - k) <= float(tol) * k:
            return True
    return False



def _pwm_neural_spectral_candidates(
    luma_field: torch.Tensor | None,
    chroma_delta_hint: torch.Tensor | None,
    *,
    full_height: int,
    min_period_px: float,
    max_period_fraction: float,
    max_candidates: int = 8,
) -> tuple[tuple[float, float], ...]:
    """Return PWM-like period candidates from neural row-correction spectra.

    Unlike autocorrelation, a mixture of two unrelated PWM sources exposes both
    fundamentals as separate Fourier peaks.  Work on the *correction* rather than
    image intensity, high-pass it fairly aggressively to reject composition-scale
    neural drift, and use the non-differentiated waveform so a square wave's
    fundamental normally outranks its edge harmonics.

    Automatic secondary-source discovery deliberately requires at least five
    visible cycles.  With fewer cycles a second frequency is not distinguishable
    enough from real illumination gradients/scene structure; users can still
    force a broad period manually through the existing period control.
    """
    if full_height < 32:
        return tuple()
    rows: list[tuple[torch.Tensor, float]] = []
    analysis_h = min(1024, int(full_height))

    def add(x: torch.Tensor | None, weight: float) -> None:
        if x is None or x.ndim != 4 or x.shape[-2] < 8 or x.shape[-1] < 2:
            return
        row = x.detach().float().median(dim=-1, keepdim=True).values
        row = _resize_row_signal(row, analysis_h)
        row = _remove_linear_y_trend(row, None)
        # About 2% of image height: enough to suppress broad scene/model drift
        # while leaving 40-80 px PWM fundamentals nearly untouched.
        row = row - _smooth_axis(row, max(6.0, 0.020 * float(analysis_h)), "y")
        for ci in range(row.shape[1]):
            vv = row[0, ci, :, 0]
            if float(vv.square().mean().sqrt()) > 2e-5:
                rows.append((vv, float(weight)))

    add(luma_field, 1.0)
    add(chroma_delta_hint, 0.40)
    if not rows:
        return tuple()

    window = torch.hann_window(analysis_h, device=rows[0][0].device, dtype=torch.float32)
    power = torch.zeros(analysis_h // 2 + 1, device=window.device, dtype=torch.float32)
    for vv, wt in rows:
        zz = torch.fft.rfft(vv * window)
        power = power + float(wt) * (zz.real.square() + zz.imag.square())

    # Secondary automatic discovery is intentionally more conservative than the
    # primary detector: require >=5 cycles.  The primary can still be much broader.
    max_full = min(
        float(full_height) * float(max_period_fraction),
        float(full_height) / 5.0,
    )
    peaks: list[tuple[float, float]] = []
    for k in range(2, int(power.numel()) - 1):
        period = float(full_height) / float(k)
        if period < float(min_period_px) or period > max_full:
            continue
        if float(power[k]) < float(power[k - 1]) or float(power[k]) < float(power[k + 1]):
            continue
        peaks.append((float(power[k]), period))
    if not peaks:
        return tuple()
    peaks.sort(key=lambda item: item[0], reverse=True)
    peak_max = max(peaks[0][0], 1e-12)
    out: list[tuple[float, float]] = []
    for pwr, period in peaks[: max(1, int(max_candidates))]:
        out.append((float(period), float(max(0.0, min(1.0, pwr / peak_max)))))
    return tuple(out)


def _pwm_discover_periods_multiregion(
    luma_field: torch.Tensor | None,
    chroma_delta_hint: torch.Tensor | None,
    y: torch.Tensor,
    *,
    full_height: int,
    primary_period: float,
    min_period_px: float,
    max_period_fraction: float,
    max_sources: int = 3,
) -> tuple[float, ...]:
    """Discover additional PWM period families in several X regions.

    A full-frame median is ideal for one dominant lamp but can hide a second LED
    that illuminates only a wall, shirt, face, etc.  This detector repeats the
    established neural/residual edge estimators on overlapping X strips, merges
    near-equal candidates, rejects small-integer harmonics of an already selected
    family, and returns at most ``max_sources`` periods.  A false extra period is
    still harmless in normal use because the downstream surface model must pass
    held-out prediction before receiving authority.

    Same-period lamps with different phases intentionally collapse to one period:
    the local sine/cosine harmonic fit can represent their different phase mixture
    on each radiometric surface without inventing a separate timing geometry.
    """
    primary = float(primary_period)
    if primary <= 2.0 or full_height < 24:
        return (primary,) if primary > 0.0 else tuple()

    # Records are [period, score, votes, neural_votes].  Period candidates from
    # each strip include the detector's selected peak and its strongest alternate
    # autocorrelation peaks.  Alternates receive a smaller vote weight.
    records: list[list[float]] = [[primary, 2.0, 3.0, 2.0]]

    def add_period(p: float, score: float, *, neural: bool, vote: float = 1.0) -> None:
        p = float(p)
        if p < float(min_period_px) or p > min(float(full_height) * float(max_period_fraction), float(full_height) / 2.0):
            return
        sc = max(0.0, float(score))
        for rec in records:
            if abs(math.log(max(p, 1e-6) / max(rec[0], 1e-6))) < math.log(1.055):
                w0 = max(1e-4, rec[1])
                w1 = max(1e-4, sc)
                rec[0] = (rec[0] * w0 + p * w1) / (w0 + w1)
                rec[1] = max(rec[1], sc)
                rec[2] += float(vote)
                if neural:
                    rec[3] += float(vote)
                return
        records.append([p, sc, float(vote), float(vote) if neural else 0.0])

    def crop_norm(x: torch.Tensor | None, lo: float, hi: float) -> torch.Tensor | None:
        if x is None or x.ndim != 4 or x.shape[-1] < 8:
            return None
        w = int(x.shape[-1])
        a = max(0, min(w - 2, int(round(float(lo) * w))))
        b = max(a + 2, min(w, int(round(float(hi) * w))))
        return x[..., a:b]

    # Global estimates are useful alternates, then five overlapping regional
    # strips make localized lamps visible.  A 42%-wide strip still contains
    # enough independent scene columns for the robust edge median to reject most
    # object boundaries.
    regions = [(0.0, 1.0)]
    width = 0.42
    for center in (0.12, 0.31, 0.50, 0.69, 0.88):
        regions.append((max(0.0, center - width / 2.0), min(1.0, center + width / 2.0)))

    for ri, (lo, hi) in enumerate(regions):
        ly = crop_norm(luma_field, lo, hi)
        cc = crop_norm(chroma_delta_hint, lo, hi)
        yy = crop_norm(y, lo, hi)

        # Fourier candidates are the important v5 addition: unlike a single
        # autocorrelation maximum they can expose two unrelated source periods
        # simultaneously.  Only candidates with meaningful relative power count
        # as full neural votes; weaker peaks are retained as fractional support.
        spectral = _pwm_neural_spectral_candidates(
            ly, cc, full_height=full_height,
            min_period_px=min_period_px,
            max_period_fraction=max_period_fraction,
            max_candidates=8,
        )
        for p, sc in spectral:
            if sc < 0.10:
                continue
            vote = 1.0 if sc >= 0.18 else 0.25
            add_period(p, sc, neural=True, vote=vote)

        # Residual autocorrelation does not create a source on its own; it only
        # reinforces a nearby neural spectral candidate during record merging.
        # This prevents a long shirt/table edge from becoming a new LED family.
        if yy is not None:
            re = _pwm_residual_edge_period(
                yy, full_height=full_height,
                min_period_px=min_period_px, max_period_fraction=max_period_fraction,
            )
            if re is not None:
                add_period(re.period_px, max(0.10, re.confidence), neural=False, vote=0.55)

    # Primary stays first/exact enough for backward compatibility.  Additional
    # periods need either repeated regional support or one strong neural vote.
    selected: list[float] = [primary]
    others = sorted(records[1:], key=lambda r: (r[1] * (1.0 + 0.12 * r[2]) + 0.08 * r[3]), reverse=True)
    for p, score, votes, neural_votes in others:
        if len(selected) >= max(1, int(max_sources)):
            break
        if any(_pwm_periods_harmonic_related(p, q) for q in selected):
            continue
        if score < 0.18:
            continue
        # Two independent/overlapping X-region neural votes are required.  This
        # is the main guard against scene-specific neural drift masquerading as a
        # second lamp.  Overlapping strips mean a real localized source normally
        # receives two votes while a single accidental boundary does not.
        if neural_votes < 1.90:
            continue
        selected.append(float(p))
    return tuple(selected)


def _pwm_multiperiod_basis(
    height: int,
    periods: tuple[float, ...],
    *,
    device: torch.device,
    max_columns: int = 20,
) -> torch.Tensor:
    """Build a de-duplicated sine/cosine basis for several PWM families."""
    if height < 4 or not periods:
        return torch.empty((0, height), device=device, dtype=torch.float32)
    freqs: list[float] = []
    # More harmonics are useful for the primary square-ish source.  Secondary
    # sources are kept somewhat lower-dimensional so a weak candidate cannot
    # overfit scene texture.
    for pi, pp in enumerate(periods):
        p = float(pp)
        if p <= 4.0:
            continue
        kmax = max(1, min(5 if pi == 0 else 4, int(max(1.0, math.floor(p / 10.0)))))
        for k in range(1, kmax + 1):
            f = float(k) / p
            if any(abs(f - old) / max(f, old, 1e-8) < 0.022 for old in freqs):
                continue
            freqs.append(f)
            if 2 * len(freqs) >= int(max_columns):
                break
        if 2 * len(freqs) >= int(max_columns):
            break
    yy = torch.arange(height, device=device, dtype=torch.float32)
    cols: list[torch.Tensor] = []
    for f in freqs:
        cols.append(torch.sin(2.0 * math.pi * f * yy))
        cols.append(torch.cos(2.0 * math.pi * f * yy))
    if not cols:
        return torch.empty((0, height), device=device, dtype=torch.float32)
    basis = torch.stack(cols, dim=0)
    rms = basis.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    return basis / rms


def _pwm_radiometric_region_masks(
    surface_guide: torch.Tensor,
    *,
    period_hint: float,
    max_regions: int = 12,
    analysis_long_side: int = 192,
) -> torch.Tensor:
    """Return soft SLIC-like radiometric regions without external dependencies.

    The guide is first made resistant to horizontal PWM state changes, then a
    deterministic low-resolution k-means uses log-luma, Cb/Cr and modest X/Y
    coordinates.  Spatial coordinates make each cluster coherent/compact while
    color/luma let real object boundaries (wall/shirt/skin/table...) dominate over
    the straight flicker stripes.  Bilinear upsampling of hard low-res labels
    gives a narrow soft boundary rather than a seam.
    """
    if surface_guide.ndim != 4 or surface_guide.shape[1] < 1:
        raise ValueError("PWM surface guide must be BCHW")
    b, _, h, w = surface_guide.shape
    long_side = max(h, w)
    scale = min(1.0, float(max(64, analysis_long_side)) / float(max(1, long_side)))
    ah = max(32, int(round(h * scale)))
    aw = max(32, int(round(w * scale)))

    guide = surface_guide.float()
    gy = _smooth_axis(guide[:, 0:1], max(3.0, float(period_hint) * 0.38), "y")
    gy = _smooth_axis(gy, 2.0, "x")
    if guide.shape[1] >= 3:
        gc = _smooth_axis(guide[:, 1:3], max(3.0, float(period_hint) * 0.38), "y")
        gc = _smooth_axis(gc, 2.0, "x")
    else:
        gc = torch.zeros((b, 2, h, w), device=guide.device, dtype=torch.float32)
    small = F.interpolate(torch.cat([gy, gc], dim=1), size=(ah, aw), mode="area")

    yy = torch.linspace(0.0, 1.0, ah, device=guide.device, dtype=torch.float32).view(1, 1, ah, 1).expand(b, -1, -1, aw)
    xx = torch.linspace(0.0, 1.0, aw, device=guide.device, dtype=torch.float32).view(1, 1, 1, aw).expand(b, -1, ah, -1)
    feat = torch.cat([
        torch.log(small[:, 0:1].clamp_min(0.0) + 0.02) / 0.34,
        small[:, 1:2] / 0.055,
        small[:, 2:3] / 0.055,
        1.45 * xx,
        1.15 * yy,
    ], dim=1)  # Bx5xahxaw

    k = max(4, min(int(max_regions), 16, int(round((ah * aw) / 2600.0))))
    # Regular-grid seed locations: deterministic and spatially well spread.
    nx = max(2, int(round(math.sqrt(k * float(aw) / float(max(1, ah))))))
    ny = max(2, int(math.ceil(float(k) / float(nx))))
    seed_xy: list[tuple[int, int]] = []
    for iy in range(ny):
        sy = int(round((iy + 0.5) * ah / ny - 0.5))
        for ix in range(nx):
            sx = int(round((ix + 0.5) * aw / nx - 0.5))
            seed_xy.append((max(0, min(ah - 1, sy)), max(0, min(aw - 1, sx))))
    seed_xy = seed_xy[:k]

    all_masks: list[torch.Tensor] = []
    flat = feat.permute(0, 2, 3, 1).reshape(b, ah * aw, 5)
    for bi in range(b):
        centers = torch.stack([feat[bi, :, sy, sx] for sy, sx in seed_xy], dim=0).float()
        labels = torch.zeros((ah * aw,), device=guide.device, dtype=torch.long)
        for _ in range(8):
            # N is only ~20k-35k, K <= 12 in normal images; this stays small.
            dist = (flat[bi, :, None, :] - centers[None, :, :]).square().sum(dim=-1)
            labels = torch.argmin(dist, dim=1)
            new_centers = []
            for ki in range(k):
                sel = labels == ki
                if bool(sel.any()):
                    new_centers.append(flat[bi, sel].mean(dim=0))
                else:
                    new_centers.append(centers[ki])
            centers = torch.stack(new_centers, dim=0)
        onehot = F.one_hot(labels, num_classes=k).to(torch.float32).transpose(0, 1).reshape(k, ah, aw)
        all_masks.append(onehot.unsqueeze(0))
    low = torch.cat(all_masks, dim=0)
    masks = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
    masks = masks.clamp_min(0.0)
    return masks / masks.sum(dim=1, keepdim=True).clamp_min(1e-6)



def _pwm_phase_lock_diagnostics(
    residual: torch.Tensor,
    support: torch.Tensor,
    *,
    period: float,
    reference_correction: torch.Tensor | None = None,
    strips: int = 4,
) -> tuple[float, float, int, float]:
    """Measure whether one phase-locked PWM source dominates the whole scene.

    This is intentionally different from the v5 segmented model.  It asks whether
    the same fundamental phase recurs across several *perpendicular* scene strips.
    A value near one means that wall, subject, clothing, etc. all see the same
    straight rolling-shutter timing.  In that case allowing every radiometric
    region to choose an independent sine/cosine phase is unnecessary freedom and
    can turn scene structure into correction artifacts.

    Returns ``(phase_coherence, median_amplitude, strip_votes, reference_ratio)``.
    ``reference_ratio`` compares neural-correction power at this period with the
    visible residual.  A small value means Restormer supplied little useful PWM
    evidence and the image itself should be the timing/amplitude authority.
    """
    if residual.ndim != 4 or residual.shape[1] != 1 or support.ndim != 4 or period <= 4.0:
        return 0.0, 0.0, 0, 0.0
    b, _, h, w = residual.shape
    if b != 1 or h < max(24, int(round(3.0 * period))) or w < 16:
        return 0.0, 0.0, 0, 0.0

    base_sigma = max(8.0, 1.8 * float(period))
    hp = residual.float() - _smooth_axis(residual.float(), base_sigma, "y")
    ww = support.float().clamp(0.0, 1.0)
    yy = torch.arange(h, device=residual.device, dtype=torch.float32).view(1, 1, h, 1)
    omega = 2.0 * math.pi / float(period)
    sn = torch.sin(yy * omega)
    cs = torch.cos(yy * omega)
    den = ww.sum(dim=-2).clamp_min(1e-6)  # Bx1xW
    aa = 2.0 * (hp * ww * sn).sum(dim=-2) / den
    bb = 2.0 * (hp * ww * cs).sum(dim=-2) / den
    amp = torch.sqrt(aa.square() + bb.square()).clamp_min(1e-8)  # Bx1xW
    cov = (den / float(h)).clamp(0.0, 1.0)
    valid = cov > 0.18
    if int(valid.sum()) < max(8, w // 12):
        return 0.0, 0.0, 0, 0.0

    phase_w = amp.square() * cov * valid.to(amp.dtype)
    sx = (phase_w * (aa / amp)).sum()
    sy = (phase_w * (bb / amp)).sum()
    sw = phase_w.sum().clamp_min(1e-8)
    coherence = float(torch.sqrt(sx.square() + sy.square()) / sw)
    global_phase = math.atan2(float(sy), float(sx))
    med_amp = float(amp[valid].median()) if bool(valid.any()) else 0.0

    # Require the same phase in several widely separated perpendicular strips.
    # This rejects a periodic wall/curtain that occupies only one part of the
    # frame while accepting true sensor-row flicker that continues across wall,
    # face, hands, clothing, etc.
    votes = 0
    nstrips = max(3, int(strips))
    for si in range(nstrips):
        x0 = int(round(si * w / nstrips))
        x1 = int(round((si + 1) * w / nstrips))
        if x1 <= x0:
            continue
        va = valid[..., x0:x1]
        if int(va.sum()) < max(3, (x1 - x0) // 10):
            continue
        aw = amp[..., x0:x1]
        pw = phase_w[..., x0:x1]
        ax = aa[..., x0:x1]
        bx = bb[..., x0:x1]
        sr = (pw * (ax / aw)).sum()
        si_im = (pw * (bx / aw)).sum()
        strip_phase = math.atan2(float(si_im), float(sr))
        dphi = abs(math.atan2(math.sin(strip_phase - global_phase), math.cos(strip_phase - global_phase)))
        strip_med = float(aw[va].median()) if bool(va.any()) else 0.0
        if dphi <= 0.38 and strip_med >= max(0.006, 0.32 * med_amp):
            votes += 1

    ref_ratio = 0.0
    if reference_correction is not None:
        ref = reference_correction.float().to(device=residual.device)
        if ref.shape[-2:] != (h, w):
            ref = F.interpolate(ref, size=(h, w), mode="bilinear", align_corners=False)
        if ref.ndim == 4 and ref.shape[1] == 1:
            rhp = ref - _smooth_axis(ref, base_sigma, "y")
            ra = 2.0 * (rhp * ww * sn).sum(dim=-2) / den
            rb = 2.0 * (rhp * ww * cs).sum(dim=-2) / den
            ramp = torch.sqrt(ra.square() + rb.square())
            if bool(valid.any()):
                ref_ratio = float(ramp[valid].median() / amp[valid].median().clamp_min(1e-6))

    return coherence, med_amp, votes, ref_ratio


def _pwm_refine_phase_locked_period(
    y: torch.Tensor,
    support: torch.Tensor,
    *,
    period: float,
    reference_correction: torch.Tensor | None = None,
) -> tuple[float, tuple[float, float, int, float]]:
    """Refine an already plausible primary period using cross-scene phase lock.

    This is a narrow refinement, not another period search.  It is enabled only
    when the candidate already produces the same fundamental phase in at least
    three perpendicular scene strips.  That lets dense PWM settle from a coarse
    autocorrelation bin (e.g. 21.50 px) to the visually coherent fundamental
    (~21.15 px) without dragging ambiguous/multi-source frames toward a scene
    texture frequency.
    """
    p0 = float(period)
    if p0 <= 4.0 or y.ndim != 4 or y.shape[1] != 1:
        return p0, (0.0, 0.0, 0, 0.0)
    residual = torch.log(y.float().clamp_min(0.0) + 0.02)
    base = _pwm_phase_lock_diagnostics(
        residual, support, period=p0, reference_correction=reference_correction,
    )
    if base[0] < 0.84 or base[2] < 3 or base[1] < 0.008:
        return p0, base

    span = max(0.20, 0.035 * p0)
    lo = max(4.0, p0 - span)
    hi = p0 + span
    best_p = p0
    best_diag = base
    best_score = base[0] * (0.70 + 0.30 * min(1.0, base[2] / 4.0)) * math.sqrt(max(base[1], 1e-8))
    # 29 samples keep sub-pixel timing stable even on high-resolution inputs.
    for i in range(29):
        pp = lo + (hi - lo) * float(i) / 28.0
        diag = _pwm_phase_lock_diagnostics(
            residual, support, period=pp, reference_correction=reference_correction,
        )
        score = diag[0] * (0.70 + 0.30 * min(1.0, diag[2] / 4.0)) * math.sqrt(max(diag[1], 1e-8))
        if diag[2] >= 3 and score > best_score:
            best_p, best_diag, best_score = float(pp), diag, float(score)
    return best_p, best_diag


def _pwm_phase_locked_local_field(
    residual: torch.Tensor,
    support: torch.Tensor,
    *,
    period: float,
    max_gain: float = 1.35,
    surface_guide: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit only local *amplitude* of one globally phase-locked PWM waveform.

    The waveform is estimated from robust cross-scene row consensus and folded
    at the detected period.  Local regions may scale that waveform, but cannot
    rotate its phase or invent independent harmonics.  This is deliberately the
    low-degree-of-freedom counterpart to the v5 segmented multi-source model.
    """
    if residual.ndim != 4 or support.ndim != 4 or support.shape[1] != 1:
        raise ValueError("phase-locked PWM inputs must be BCHW with scalar support")
    b, c, h, w = residual.shape
    if period <= 4.0 or h < max(24, int(round(3.0 * period))):
        z = torch.zeros_like(residual)
        e = torch.zeros((b, 1, h, w), device=residual.device, dtype=residual.dtype)
        return z, e, e

    base_sigma = max(8.0, 1.8 * float(period))
    # Correction sign: target is the negative of the visible band-scale residual.
    target = _smooth_axis(residual.float(), base_sigma, "y") - residual.float()
    ww = support.float().clamp(0.0, 1.0)
    row, valid, conf, _ = _robust_row_consensus(target, ww, huber_k=2.5, min_coverage=0.08)
    row = _fill_row_profile(row, valid, max(1.0, 0.05 * float(period)))

    bins = max(24, min(384, int(round(float(period) * 4.0))))
    yy = torch.arange(h, device=residual.device, dtype=torch.float32)
    idx = torch.floor(torch.remainder(yy / float(period), 1.0) * bins).long().clamp(0, bins - 1)
    idx_bc = idx.view(1, 1, h).expand(b, c, -1)
    rowv = row[..., 0]
    rw = (valid * conf).expand(-1, c, -1, -1)[..., 0].clamp(0.0, 1.0)
    num = torch.zeros((b, c, bins), device=residual.device, dtype=torch.float32)
    den = torch.zeros_like(num)
    num.scatter_add_(2, idx_bc, rowv * rw)
    den.scatter_add_(2, idx_bc, rw)
    fold = num / den.clamp_min(1e-6)
    fold = _circular_smooth_profile(fold.reshape(b * c, bins), 0.8).reshape(b, c, bins)
    fold = fold - fold.mean(dim=-1, keepdim=True)
    template = torch.gather(fold, 2, idx_bc).unsqueeze(-1)  # BxCxHx1
    q = template.expand(-1, -1, -1, w)

    # Phase is fixed; only a positive slowly-varying scalar gain is fitted.
    # Wide support along the band axis averages several complete cycles, while a
    # much shorter perpendicular support lets wall/skin/clothing amplitudes differ.
    sup = ww.expand(-1, c, -1, -1)
    num2 = target * q * sup
    den2 = q.square() * sup
    pow2 = target.square() * sup
    sy = max(12.0, 3.2 * float(period))
    sx = max(10.0, min(48.0, 0.015 * float(w)))
    for axis, sigma in (("y", sy), ("x", sx)):
        num2 = _smooth_axis(num2, sigma, axis)
        den2 = _smooth_axis(den2, sigma, axis)
        pow2 = _smooth_axis(pow2, sigma, axis)
    gain = (num2 / den2.clamp_min(1e-7)).clamp(0.0, float(max_gain))
    gain = _smooth_axis(gain, max(4.0, 0.35 * float(period)), "y")
    gain = _smooth_axis(gain, max(3.0, 0.35 * sx), "x")

    corr = num2 / torch.sqrt((den2 * pow2).clamp_min(1e-10))
    corr_gate = 0.55 + 0.45 * _smoothstep(corr, 0.16, 0.55)

    # Surface-conditioned scalar gains.  Unlike v5 segmentation, a region is not
    # allowed to choose phase or harmonic shape; it can only scale the globally
    # locked waveform.  Even/odd-cycle validation rejects a region whose apparent
    # gain comes from scene structure rather than repeatable PWM.
    if surface_guide is not None and surface_guide.ndim == 4 and surface_guide.shape[-2:] == (h, w):
        masks = _pwm_radiometric_region_masks(
            surface_guide.float(), period_hint=float(period), max_regions=12, analysis_long_side=176,
        )
        reg_gain = torch.zeros_like(gain)
        reg_auth = torch.zeros((b, 1, h, w), device=residual.device, dtype=torch.float32)
        cyc = torch.floor(torch.arange(h, device=residual.device, dtype=torch.float32) / float(period)).long()
        even = ((cyc % 2) == 0).view(1, 1, h, 1).to(torch.float32)
        odd = 1.0 - even
        for ri in range(masks.shape[1]):
            rm = masks[:, ri:ri+1].float() * ww
            mass = float(rm.mean())
            if mass < 0.025:
                continue
            # Need broad band-axis coverage so one object edge cannot define gain.
            row_cov = (rm.mean(dim=-1, keepdim=True) > 0.06).float().mean()
            if float(row_cov) < 0.18:
                continue
            rc = rm.expand(-1, c, -1, -1)
            gains = []
            improves = []
            for train, val in ((even, odd), (odd, even)):
                tr = rc * train
                va = rc * val
                den_g = (q.square() * tr).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                g = ((target * q * tr).sum(dim=(-2, -1), keepdim=True) / den_g).clamp(0.0, float(max_gain))
                gains.append(g)
                base_err = (target.square() * va).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                pred_err = ((target - q * g).square() * va).sum(dim=(-2, -1), keepdim=True)
                improves.append((1.0 - pred_err / base_err).clamp(-1.0, 1.0))
            g0, g1 = gains
            gmean = 0.5 * (g0 + g1)
            agree = 1.0 - (g0 - g1).abs() / (g0.abs() + g1.abs() + 0.10)
            improve = torch.minimum(improves[0], improves[1])
            auth_c = _smoothstep(improve, 0.015, 0.18) * _smoothstep(agree, 0.45, 0.80)
            # Scalar authority for the region, shared spatially but channel-specific gain.
            auth_s = auth_c.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
            reg_gain = reg_gain + rm * gmean * auth_s
            reg_auth = reg_auth + rm * auth_s
        reg_auth = reg_auth.clamp(0.0, 1.0)
        # Soft masks sum to one; normalized regional gain avoids boundary dimming.
        rg = reg_gain / reg_auth.expand(-1, c, -1, -1).clamp_min(1e-6)
        gain = gain * (1.0 - reg_auth) + rg * reg_auth
        corr_gate = torch.maximum(corr_gate, 0.70 * reg_auth)

    field = q * gain * corr_gate
    ev = corr_gate.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
    amp = gain.mean(dim=1, keepdim=True).clamp(0.0, float(max_gain))
    return field.to(residual.dtype), amp.to(residual.dtype), ev.to(residual.dtype)



def _pwm_polish_mode_energy(
    signal: torch.Tensor,
    support: torch.Tensor,
    *,
    period: float,
    harmonics: int = 5,
    strips: int = 4,
) -> float:
    """Return robust energy confined to one already-known PWM period family.

    The score is intentionally diagnostic rather than a new detector.  Several
    perpendicular strips are phase-projected independently and their median is
    used, so a scene edge in one part of the frame cannot make the polish stage
    believe that the known PWM mode improved everywhere.
    """
    if signal.ndim != 4 or support.ndim != 4 or support.shape[1] != 1:
        return 0.0
    b, c, h, w = signal.shape
    if b != 1 or period <= 4.0 or h < max(24, int(round(2.5 * float(period)))) or w < 12:
        return 0.0
    # Validation needs cross-scene coverage, not full perpendicular resolution.
    # Compress only the direction *along* the bands; the phase axis and period
    # remain exact.  This makes the multi-candidate line search cheap enough for
    # full-resolution GUI use without changing the measured PWM frequency.
    sig = signal.float()
    sup = support.float().clamp(0.0, 1.0)
    if w > 320:
        sig = F.interpolate(sig, size=(h, 320), mode="area")
        sup = F.interpolate(sup, size=(h, 320), mode="area")
        w = 320
    # Remove only structure much broader than the PWM family.  Projection below
    # is at exact known frequencies, so this is not a period search.
    hp = sig - _smooth_axis(sig, max(10.0, 1.8 * float(period)), "y")
    yy = torch.arange(h, device=signal.device, dtype=torch.float32).view(1, 1, h, 1)
    vals = []
    nstrips = max(3, int(strips))
    for si in range(nstrips):
        x0 = int(round(si * w / nstrips))
        x1 = int(round((si + 1) * w / nstrips))
        if x1 - x0 < 3:
            continue
        ss = sup[..., x0:x1]
        # Require useful coverage before letting a strip vote.
        if float((ss > 0.20).float().mean()) < 0.12:
            continue
        rr = hp[..., x0:x1]
        den = ss.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        e2 = torch.zeros((1,), device=signal.device, dtype=torch.float32)
        max_h = max(1, min(int(harmonics), 7))
        for k in range(1, max_h + 1):
            omega = 2.0 * math.pi * float(k) / float(period)
            sn = torch.sin(yy * omega)
            cs = torch.cos(yy * omega)
            aa = 2.0 * (rr * ss * sn).sum(dim=(-2, -1), keepdim=True) / den
            bb = 2.0 * (rr * ss * cs).sum(dim=(-2, -1), keepdim=True) / den
            # Mean over channels; sum harmonic powers so square-ish PWM remains
            # visible to the score rather than only its fundamental.
            e2 = e2 + (aa.square() + bb.square()).mean().reshape(1)
        vals.append(torch.sqrt(e2.clamp_min(0.0)))
    if not vals:
        return 0.0
    vv = torch.stack(vals).flatten()
    return float(vv.median())


def _pwm_residual_polish(
    y: torch.Tensor,
    c: torch.Tensor,
    *,
    periods: tuple[float, ...],
    support: torch.Tensor,
    edge_support: torch.Tensor,
    user_mask: torch.Tensor | None = None,
    luma_strength: float = 1.0,
    chroma_strength: float = 1.0,
    max_passes: int = 2,
    coh_stop: float = 0.85,
    eps: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, float, float, int, float, torch.Tensor, torch.Tensor]:
    """Optional final projection onto already-validated PWM timing families.

    This stage deliberately performs *no new period search*.  For each period
    accepted by the main PWM estimator it reconstructs the residual waveform at
    that fixed sensor phase, fits only slowly varying positive local amplitude,
    and uses an exact-mode line search to accept a pass only when the known PWM
    energy falls without a material rise in nearby control frequencies.

    The result is therefore a conservative residual eraser rather than another
    free-form adaptive filter.  It is especially useful after the v6 phase-lock
    path, where the remaining bands are weak copies of a timing family we already
    trust.
    """
    periods = tuple(float(p) for p in periods if float(p) > 4.0)
    if not periods or max(luma_strength, chroma_strength) <= 0.0 or int(max_passes) <= 0:
        return y, c, 0.0, 0.0, 0, 0.0, torch.zeros_like(y), torch.zeros_like(c)

    cur_y = y.float()
    cur_c = c.float()
    total_log_y = torch.zeros_like(cur_y)
    total_dc = torch.zeros_like(cur_c)
    base_support = support.float().clamp(0.0, 1.0)
    if user_mask is not None:
        um = user_mask.float().clamp(0.0, 1.0)
        if um.ndim != 4 or um.shape[1] != 1 or um.shape[-2:] != y.shape[-2:]:
            raise ValueError("PWM polish user mask must be Bx1xHxW and match Y")
        base_support = base_support * um
    else:
        um = torch.ones_like(cur_y)

    # Keep this post stage physically bounded.  1.0 means subtract the measured
    # residual; values above one are mild overdrive, not a second 2x-style gain.
    base_y_strength = max(0.0, min(1.25, float(luma_strength)))
    base_c_strength = max(0.0, min(1.25, float(chroma_strength)))
    passes_done = 0
    first_energy = None
    last_energy = None

    for _pass in range(max(1, min(6, int(max_passes)))):
        # A slow signal-level gate protects deep shadows while preserving the
        # straight band geometry across real scene/object boundaries.
        p_ref = float(min(periods))
        signal = _smooth_axis(cur_y, max(2.0, 0.40 * p_ref), "y")
        signal = _smooth_axis(signal, 2.0, "x")
        rel_y = _smoothstep(signal, 0.018, 0.070) * um
        rel_c = _smoothstep(signal, 0.028, 0.095) * um
        # PWM transitions are real illumination evidence, so the ordinary
        # edge/tone profile support is too restrictive for a final phase-locked
        # projection (it can erase exactly the rows crossing a face or jacket).
        # Use only a slow SNR gate plus the explicit user mask.  Cross-strip
        # phase coherence and held-out energy validation provide the scene-edge
        # rejection here.
        polish_support = (_smoothstep(signal, 0.008, 0.035) * um).clamp(0.0, 1.0)

        logy = torch.log(cur_y.clamp_min(0.0) + float(eps))
        logy_res = _remove_linear_y_trend(logy, polish_support)
        c_res = _remove_linear_y_trend(cur_c, polish_support)
        surface_guide = torch.cat([cur_y, cur_c], dim=1)

        sum_y = torch.zeros_like(cur_y)
        sum_c = torch.zeros_like(cur_c)
        active = []
        # P9: best phase coherence seen this pass. Once the residual stops being
        # phase-consistent with the band, further passes fit noise and overshoot.
        pass_coh = 0.0

        # Estimate the residual waveform on a compressed *along-band* grid.
        # The phase axis is never resampled, so period/phase remain exact while
        # the expensive local surface fit becomes several times faster.
        h0, w0 = cur_y.shape[-2:]
        fit_w = min(w0, 384)
        if fit_w < w0:
            fit_logy_res = F.interpolate(logy_res, size=(h0, fit_w), mode="area")
            fit_c_res = F.interpolate(c_res, size=(h0, fit_w), mode="area")
            fit_support = F.interpolate(polish_support, size=(h0, fit_w), mode="area")
            fit_guide = F.interpolate(surface_guide, size=(h0, fit_w), mode="area")
        else:
            fit_logy_res, fit_c_res = logy_res, c_res
            fit_support, fit_guide = polish_support, surface_guide

        for p0 in periods:
            # The polish path is allowed to act only when the *remaining* image
            # still shows the same phase over several separated strips.  This is
            # an anti-artifact gate, not a new timing estimator.
            coh, amp, votes, _ = _pwm_phase_lock_diagnostics(
                fit_logy_res, fit_support, period=float(p0), reference_correction=None,
            )
            if coh < 0.62 or votes < 3 or amp < 0.0012:
                print(f"[polish] p={float(p0):.3f} REJECTED  "
                      f"coh={coh:.3f}(need>=0.62)  votes={votes}(need>=3)  "
                      f"amp={amp:.6f}(need>=0.0012)")
                continue
            print(f"[polish] p={float(p0):.3f} accepted  "
                  f"coh={coh:.3f} votes={votes} amp={amp:.6f}")
            pass_coh = max(pass_coh, float(coh))
            fy, _, evy = _pwm_phase_locked_local_field(
                fit_logy_res, fit_support, period=float(p0), max_gain=1.55,
                surface_guide=fit_guide,
            )
            fc, _, evc = _pwm_phase_locked_local_field(
                fit_c_res, fit_support, period=float(p0), max_gain=1.55,
                surface_guide=fit_guide,
            )
            if fit_w < w0:
                fy = F.interpolate(fy, size=(h0, w0), mode="bilinear", align_corners=False)
                fc = F.interpolate(fc, size=(h0, w0), mode="bilinear", align_corners=False)
                evy = F.interpolate(evy, size=(h0, w0), mode="bilinear", align_corners=False)
                evc = F.interpolate(evc, size=(h0, w0), mode="bilinear", align_corners=False)

            # At this final stage the stronger safety tests are phase coherence
            # across separated strips plus the exact-mode/control-frequency line
            # search below.  A hard scene-edge no-harm map would systematically
            # leave bands on faces, hands and clothing, which is precisely what
            # this optional polish is meant to remove.
            gate_y = (rel_y * (0.70 + 0.30 * evy)).clamp(0.0, 1.0)
            gate_c = (rel_c * (0.70 + 0.30 * evc)).clamp(0.0, 1.0).expand(-1, 2, -1, -1)
            sum_y = sum_y + gate_y * fy
            sum_c = sum_c + gate_c * fc
            active.append(float(p0))

        if not active:
            print(f"[polish] pass {passes_done + 1}: no period passed the phase-lock "
                  f"gate; stopping")
            break

        # P9: phase-based stop. |energy| never reaches zero because scene texture
        # lives at the band frequency too, so it is not a usable convergence
        # signal. Falling coherence is: it says the residual is no longer a band.
        if passes_done > 0 and pass_coh < float(coh_stop):
            print(f"[polish] pass {passes_done + 1}: coherence {pass_coh:.3f} < "
                  f"{float(coh_stop):.2f}; residual is no longer a band, stopping")
            break

        before_y = sum(_pwm_polish_mode_energy(logy, polish_support, period=p) for p in active)
        before_c = sum(_pwm_polish_mode_energy(cur_c, polish_support, period=p) for p in active)
        before = before_y + 0.35 * before_c
        if first_energy is None:
            first_energy = before
        if before <= 1e-6:
            break

        # Nearby frequencies are controls.  A genuine cleanup should not make
        # them jump merely to erase the score at the exact PWM bins.
        controls = []
        for p0 in active:
            for ratio in (0.87, 1.13):
                q = p0 * ratio
                if q > 4.0 and all(abs(q / pp - 1.0) > 0.08 for pp in active):
                    controls.append(q)
        control_before = sum(_pwm_polish_mode_energy(logy, polish_support, period=q, harmonics=3) for q in controls)

        best = None
        # Preserve highlight headroom exactly as the main profile stage does.
        bright = _smoothstep(cur_y, 0.70, 0.90)
        maxy = cur_y + (1.0 - 0.45 * bright) * (1.0 - cur_y)
        maxpos = torch.log((maxy + float(eps)).clamp_min(1e-6) / (cur_y + float(eps)).clamp_min(1e-6))
        # The folded robust estimator intentionally under-reacts to strong scene
        # structure, so its raw amplitude can be conservative.  Search a bounded
        # calibration factor around it and let exact-mode validation choose the
        # actual amount.  This is not blind overdrive: candidates that invert the
        # PWM mode simply score worse and are rejected.
        for fraction in (2.00, 1.70, 1.40, 1.10, 0.85, 0.60, 0.35):
            sy = fraction * base_y_strength
            sc = fraction * base_c_strength
            dy = sum_y * sy
            dc = sum_c * sc
            dy = torch.where(dy > 0.0, torch.minimum(dy, maxpos), dy)
            # Because cy+eps == (cur_y+eps)*exp(dy), candidate log-luminance is
            # exactly logy+dy.  Score in that domain and materialize RGB-like Y
            # only once for the winning candidate.
            clog = logy + dy
            cc = cur_c + dc
            after_y = sum(_pwm_polish_mode_energy(clog, polish_support, period=p) for p in active)
            after_c = sum(_pwm_polish_mode_energy(cc, polish_support, period=p) for p in active)
            after = after_y + 0.35 * after_c
            control_after = sum(_pwm_polish_mode_energy(clog, polish_support, period=q, harmonics=3) for q in controls)
            control_ok = (not controls) or control_after <= control_before * 1.15 + 3e-4
            improve = (before - after) / max(before, 1e-8)
            if control_ok and improve >= 0.012:
                if best is None or after < best[0]:
                    best = (after, dy, dc, improve, fraction)

        if best is None:
            print(f"[polish] pass {passes_done + 1}: no candidate met "
                  f"control_ok and improve>=0.012 (before={before:.6g})")
            break
        print(f"[polish] pass {passes_done + 1}: ACCEPTED improve={best[3]:.4f} "
              f"fraction={best[4]:.2f} (early-exit if improve<0.035)")
        after, dy, dc, improve, _fraction = best
        cur_y = (cur_y + float(eps)) * torch.exp(dy) - float(eps)
        cur_c = cur_c + dc
        total_log_y = total_log_y + dy
        total_dc = total_dc + dc
        passes_done += 1
        last_energy = after
        # Once a pass only shaves a few percent from the exact known modes, a
        # further iteration is more likely to chase scene coincidences than PWM.
        if improve < 0.035:
            break

    if passes_done <= 0:
        return y, c, 0.0, 0.0, 0, 0.0, total_log_y, total_dc
    yrms = float(total_log_y.square().mean().sqrt())
    crms = float(total_dc.square().mean().sqrt())
    overall = 0.0
    if first_energy is not None and last_energy is not None and first_energy > 1e-8:
        overall = max(0.0, min(1.0, (first_energy - last_energy) / first_energy))
    return cur_y, cur_c, yrms, crms, passes_done, overall, total_log_y, total_dc

def _pwm_segmented_multisource_field(
    residual: torch.Tensor,
    support: torch.Tensor,
    *,
    periods: tuple[float, ...],
    baseline_field: torch.Tensor,
    fallback_field: torch.Tensor | None = None,
    surface_guide: torch.Tensor,
    reference_correction: torch.Tensor | None = None,
    max_regions: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refine a PWM field on radiometrically coherent held-out segments.

    ``baseline_field`` is the conservative v3/v4 global result.  Each segment may
    fit a mixture of all detected period/harmonic quadratures, so different LED
    sources and same-frequency/different-phase mixtures can have different local
    coefficients.  The segment never receives authority merely because it fits
    its training pixels: alternating X blocks are held out, and the local model
    must beat the existing baseline on *both* folds.  Weak/ambiguous segments
    therefore fall back exactly to the established global correction.
    """
    if residual.ndim != 4 or support.ndim != 4 or support.shape[1] != 1:
        raise ValueError("segmented PWM inputs must be BCHW with scalar support")
    if baseline_field.shape != residual.shape:
        raise ValueError("segmented PWM baseline field must match residual")
    if fallback_field is not None and fallback_field.shape != residual.shape:
        raise ValueError("segmented PWM fallback field must match residual")
    b, c, h, w = residual.shape
    if not periods or h < 24 or w < 16:
        z = torch.zeros((b, 1, h, w), device=residual.device, dtype=residual.dtype)
        return baseline_field, z, z

    basis = _pwm_multiperiod_basis(h, periods, device=residual.device)
    if basis.shape[0] < 2:
        z = torch.zeros((b, 1, h, w), device=residual.device, dtype=residual.dtype)
        return baseline_field, z, z
    # HxJ for the small normal equations below.
    bhj = basis.transpose(0, 1).contiguous()
    jn = int(bhj.shape[1])

    base_sigma = max(12.0, min(float(h) / 7.0, max(float(p) for p in periods) * 1.25))
    target = _smooth_axis(residual.float(), base_sigma, "y") - residual.float()
    masks = _pwm_radiometric_region_masks(
        surface_guide, period_hint=float(periods[0]), max_regions=max_regions,
    ).to(device=residual.device, dtype=torch.float32)
    k_regions = int(masks.shape[1])
    sup = support.float().clamp(0.0, 1.0)

    # Alternating broad X blocks create a much stronger test than fitting/validating
    # different rows: true rolling-shutter bands repeat at the same Y on unseen X,
    # while most object texture/edges do not.
    block = max(24, min(96, int(round(float(w) / 32.0))))
    xidx = torch.arange(w, device=residual.device)
    fold0 = (((xidx // block) % 2) == 0).to(torch.float32).view(1, 1, 1, w)
    fold1 = 1.0 - fold0

    ref_band = None
    if reference_correction is not None:
        rr = reference_correction.float().to(device=residual.device)
        if rr.shape[-2:] != (h, w):
            rr = F.interpolate(rr, size=(h, w), mode="bilinear", align_corners=False)
        if rr.shape[1] == c:
            ref_band = rr - _smooth_axis(rr, base_sigma, "y")

    result = baseline_field.float().clone()
    authority_map = torch.zeros((b, 1, h, w), device=residual.device, dtype=torch.float32)

    def row_profile(x: torch.Tensor, ww: torch.Tensor, ci: int) -> tuple[torch.Tensor, torch.Tensor]:
        den = ww.sum(dim=-1)[0, 0]  # H
        num = (x[0, ci] * ww[0, 0]).sum(dim=-1)
        return num / den.clamp_min(1e-6), den

    def solve_coeff(profile: torch.Tensor, den: torch.Tensor, prior: torch.Tensor | None, ridge_ratio: float) -> torch.Tensor:
        row_w = torch.minimum(den, torch.full_like(den, 256.0)) * (den > 3.0).to(den.dtype)
        wb = bhj * row_w[:, None]
        gram = bhj.transpose(0, 1) @ wb
        scale = torch.trace(gram) / max(1, jn)
        lam = float(ridge_ratio) * scale.clamp_min(1e-8)
        gram = gram + torch.eye(jn, device=gram.device, dtype=gram.dtype) * lam
        rhs = bhj.transpose(0, 1) @ (profile * row_w)
        if prior is not None:
            rhs = rhs + lam * prior
        try:
            return torch.linalg.solve(gram, rhs)
        except RuntimeError:
            return torch.linalg.lstsq(gram, rhs.unsqueeze(1)).solution[:, 0]

    def weighted_error(profile: torch.Tensor, pred: torch.Tensor, den: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        row_w = torch.minimum(den, torch.full_like(den, 256.0)) * (den > 3.0).to(den.dtype)
        err = ((profile - pred).square() * row_w).sum()
        power = (profile.square() * row_w).sum().clamp_min(1e-10)
        return err, power

    # Current implementation, like the rest of this cleanup stage, is optimized
    # for batch=1 inference.  For larger batches simply process each sample by
    # recursion so region memberships remain sample-specific.
    if b != 1:
        outs = []
        auths = []
        for bi in range(b):
            out_i, a_i, _ = _pwm_segmented_multisource_field(
                residual[bi:bi+1], support[bi:bi+1], periods=periods,
                baseline_field=baseline_field[bi:bi+1],
                fallback_field=(None if fallback_field is None else fallback_field[bi:bi+1]),
                surface_guide=surface_guide[bi:bi+1],
                reference_correction=(None if reference_correction is None else reference_correction[bi:bi+1]),
                max_regions=max_regions,
            )
            outs.append(out_i)
            auths.append(a_i)
        out = torch.cat(outs, dim=0)
        auth = torch.cat(auths, dim=0)
        return out, auth, auth

    for ki in range(k_regions):
        region = masks[:, ki:ki+1]
        region_support = (region * sup).clamp(0.0, 1.0)
        if float(region_support.mean()) < 0.012:
            continue
        # First choose between the established v4 surface baseline and the more
        # conservative global/cycle-consensus fallback.  This explicitly lets a
        # coherent region back away from a v4 local artifact before any new
        # harmonic freedom is introduced.  The decision is held out in X.
        fallback_authority = 0.0
        if fallback_field is not None:
            fb_fold_adv: list[float] = []
            for valid_fold in (fold0, fold1):
                wvalid = region_support * valid_fold
                pval, dval = row_profile(target, wvalid, 0)
                pbase, _ = row_profile(baseline_field.float(), wvalid, 0)
                pfb, _ = row_profile(fallback_field.float(), wvalid, 0)
                ebase, power = weighted_error(pval, pbase, dval)
                efb, _ = weighted_error(pval, pfb, dval)
                fb_fold_adv.append(float(((ebase - efb) / power).clamp(-1.0, 1.0)))
            fb_adv = min(fb_fold_adv)
            if fb_adv > 0.0015:
                fallback_authority = min(0.92, float(_smoothstep(
                    torch.tensor(fb_adv, device=residual.device), 0.0015, 0.026
                )))
                for ci in range(c):
                    result[:, ci:ci+1] = result[:, ci:ci+1] + region * fallback_authority * (
                        fallback_field[:, ci:ci+1].float() - baseline_field[:, ci:ci+1].float()
                    )
                authority_map = torch.maximum(authority_map, region * (0.55 * fallback_authority))

        # One scalar harmonic-refinement authority per region; channel-specific
        # coefficients are still allowed.  Luma determines most of the safety
        # decision, chroma can only increase confidence modestly through reference
        # presence.
        region_advantages: list[float] = []
        full_coeffs: list[torch.Tensor] = []
        local_rows: list[torch.Tensor] = []
        for ci in range(c):
            # Prior is the existing baseline projected onto the same nuisance
            # basis.  The local model is therefore a refinement, not a restart.
            selected_base_field = baseline_field.float()
            if fallback_field is not None and fallback_authority > 0.0:
                selected_base_field = (
                    baseline_field.float() * (1.0 - fallback_authority)
                    + fallback_field.float() * fallback_authority
                )
            pbase, dbase = row_profile(selected_base_field, region_support, ci)
            prior = solve_coeff(pbase, dbase, None, 0.04)
            fold_adv: list[float] = []
            fold_benefit: list[float] = []
            for train_fold, valid_fold in ((fold0, fold1), (fold1, fold0)):
                wtrain = region_support * train_fold
                wvalid = region_support * valid_fold
                ptr, dtr = row_profile(target, wtrain, ci)
                coeff = solve_coeff(ptr, dtr, prior, 0.34)
                pred = bhj @ coeff
                pval, dval = row_profile(target, wvalid, ci)
                pbaseline, _ = row_profile(selected_base_field, wvalid, ci)
                e_local, power = weighted_error(pval, pred, dval)
                e_base, _ = weighted_error(pval, pbaseline, dval)
                e_zero, _ = weighted_error(pval, torch.zeros_like(pval), dval)
                fold_adv.append(float(((e_base - e_local) / power).clamp(-1.0, 1.0)))
                fold_benefit.append(float(((e_zero - e_local) / power).clamp(-1.0, 1.0)))
            adv = min(fold_adv)
            benefit = min(fold_benefit)
            # Luma is the main validator.  Chroma is noisier, so only count a
            # positive chroma vote and never let it rescue a failed luma segment.
            if ci == 0:
                region_advantages.append(min(adv, benefit))
            else:
                region_advantages.append(max(-0.02, min(adv, benefit)))

            pfull, dfull = row_profile(target, region_support, ci)
            coeff_full = solve_coeff(pfull, dfull, prior, 0.30)
            # Prevent a tiny/odd segment from exploding a harmonic coefficient.
            prior_rms = prior.square().mean().sqrt().clamp_min(2e-4)
            coeff_rms = coeff_full.square().mean().sqrt().clamp_min(1e-8)
            cap = 3.0 * prior_rms + (0.010 if c == 1 else 0.006)
            if float(coeff_rms) > float(cap):
                coeff_full = coeff_full * (cap / coeff_rms)
            full_coeffs.append(coeff_full)
            local_rows.append(bhj @ coeff_full)

        if not region_advantages:
            continue
        luma_score = float(region_advantages[0])
        if luma_score <= 0.0:
            continue
        authority = float(_smoothstep(torch.tensor(luma_score, device=residual.device), 0.0035, 0.040))
        authority = min(0.78, max(0.0, authority))
        if authority <= 1e-4:
            continue

        # A real PWM source should usually leave at least a small trace in the
        # cumulative neural correction.  Presence only; sign disagreement is not
        # a veto because the residual may need to undo local neural over-correction.
        if ref_band is not None:
            ref_prof, ref_den = row_profile(ref_band, region_support, 0)
            ref_coeff = solve_coeff(ref_prof, ref_den, None, 0.08)
            ref_energy = float(ref_coeff.square().mean().sqrt())
            ref_gate = float(_smoothstep(torch.tensor(ref_energy, device=residual.device), 7e-5, 7e-4))
            authority *= 0.45 + 0.55 * ref_gate

        for ci in range(c):
            local = local_rows[ci].view(1, 1, h, 1).expand(1, 1, h, w)
            base_ch = selected_base_field[:, ci:ci+1].float()
            result[:, ci:ci+1] = result[:, ci:ci+1] + region * authority * (local - base_ch)
        authority_map = torch.maximum(authority_map, region * authority)

    # Soft region masks already feather boundaries.  A very small final feather
    # removes low-resolution label stair-steps without letting one surface smear
    # broadly into another.
    authority_map = _smooth_axis(_smooth_axis(authority_map, 1.25, "x"), 1.0, "y").clamp(0.0, 1.0)
    return result.to(residual.dtype), authority_map.to(residual.dtype), authority_map.to(residual.dtype)


def _pwm_unit_step_template(
    *,
    batch: int,
    height: int,
    period: float,
    transition_phases: tuple[float, float, float],
    transition_ratio: float,
    device,
    dtype,
) -> torch.Tensor:
    """Build a zero-DC, unit-contrast PWM state template in Bx1xHx1."""
    bins = max(16, min(768, int(round(float(period)))))
    p1 = int(round((transition_phases[0] % 1.0) * bins)) % bins
    p2 = int(round((transition_phases[1] % 1.0) * bins)) % bins
    lo, hi = sorted((p1, p2))
    idx = torch.arange(bins, device=device)
    state = (idx > lo) & (idx <= hi)
    if int(state.sum()) < 2 or int((~state).sum()) < 2:
        raise ValueError("Degenerate PWM transition phases")
    phase = state.to(dtype).view(1, -1)
    phase = phase - phase.mean(dim=-1, keepdim=True)
    # Real PWM/strobe transitions are sharp compared with the plateau, but they
    # are not mathematically one-row steps after sensor integration, demosaicing
    # and the neural pass.  A small finite rise/fall also makes the estimator less
    # sensitive to a 1-3 row phase discrepancy between the neural edge hint and
    # the visible residual.  Keep the existing transition_ratio as a lower bound
    # while guaranteeing roughly a few-bin transition at ordinary periods.
    # Keep the configured transition ratio authoritative.  The prototype that
    # performed best on hard PWM/strobe bands used genuinely sharp state edges;
    # forcing a 3%-of-period minimum rounded the boundary and left a repeated
    # thin residual stripe.
    sigma = max(0.30, min(2.25, max(float(transition_ratio) * bins, 1.40)))
    phase = _circular_smooth_profile(phase, sigma)
    a = phase[0, state].median()
    b = phase[0, ~state].median()
    phase = phase / (a - b).abs().clamp_min(1e-6)

    yy = torch.arange(height, device=device, dtype=torch.float32)
    phase_idx = torch.floor(torch.remainder(yy / float(period), 1.0) * bins).long().clamp(0, bins - 1)
    row = phase[:, phase_idx].view(1, 1, height, 1)
    return row.expand(batch, -1, -1, -1).to(dtype=dtype)


def _pwm_xonly_direct_field(
    residual: torch.Tensor,
    template: torch.Tensor,
    support: torch.Tensor,
    edge_support: torch.Tensor,
    *,
    period: float,
    sigma_x_ratio: float,
    corr_low: float = 0.005,
    corr_high: float = 0.045,
    use_edge_support: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit independent channel amplitudes for one shared edge-locked template.

    The shared template fixes period/phase/duty.  Y, Cb and Cr may have different
    signed amplitudes across X because surface spectra differ under the same LED
    states.  Coefficients are smoothed only across X; there is no Y-local gain.
    """
    if template.ndim != 4 or template.shape[1] != 1 or template.shape[-1] != 1:
        raise ValueError("PWM template must be Bx1xHx1")
    q = template.expand(-1, residual.shape[1], -1, residual.shape[-1]).float()
    w = support.float().clamp(0.0, 1.0)
    if use_edge_support:
        w = (w * edge_support.float().clamp(0.0, 1.0)).clamp(0.0, 1.0)
    wc = w.expand(-1, residual.shape[1], -1, -1)

    base_sigma = max(12.0, float(period) * 1.25)
    target = _smooth_axis(residual.float(), base_sigma, "y") - residual.float()
    num = (target * q * wc).sum(dim=-2, keepdim=True)
    den = (q.square() * wc).sum(dim=-2, keepdim=True)
    power = (target.square() * wc).sum(dim=-2, keepdim=True)
    coverage = w.mean(dim=-2, keepdim=True)

    prototype_min_x = min(128.0, max(48.0, 0.064 * float(residual.shape[-1])))
    sx = max(prototype_min_x, min(256.0, float(period) * max(0.75, float(sigma_x_ratio))))
    num = _smooth_axis(num, sx, "x")
    den = _smooth_axis(den, sx, "x")
    power = _smooth_axis(power, sx, "x")
    coverage = _smooth_axis(coverage, sx, "x")

    coeff = num / den.clamp_min(1e-8)
    signed_corr = num / torch.sqrt((den * power).clamp_min(1e-12))
    evidence = _smoothstep(signed_corr.abs(), float(corr_low), float(corr_high))
    evidence = evidence * _smoothstep(coverage, 0.03, 0.18)

    # Robustly cap isolated coefficient spikes without constraining the normal
    # fitted amplitude.  The user's Profile strength remains the public scalar.
    abs_coeff = coeff.abs()
    ref = torch.quantile(abs_coeff, 0.80, dim=-1, keepdim=True).clamp_min(1e-5)
    limit = 3.0 * ref
    coeff = torch.maximum(torch.minimum(coeff, limit), -limit)

    field = q * coeff.expand(-1, -1, residual.shape[-2], -1) * evidence.expand(-1, -1, residual.shape[-2], -1)
    # Public/debug amplitude maps remain scalar; channel-specific coefficients
    # are already baked into ``field``.
    ev_scalar = evidence.mean(dim=1, keepdim=True).expand(-1, 1, residual.shape[-2], -1)
    amp_scalar = ev_scalar
    return field, amp_scalar, ev_scalar



def _pwm_cycle_consensus_field(
    residual: torch.Tensor,
    support: torch.Tensor,
    *,
    period: float,
    sigma_x_ratio: float,
    max_harmonics: int = 5,
    reference_correction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate repeatable PWM residuals with cycle-consensus harmonics.

    A post-neural square-wave artifact is rarely an ideal two-level step: the
    network often removes most of each plateau but leaves rounded shoulders,
    ringing and a few stable harmonics.  Fitting one square template therefore
    leaves those residuals behind or over-corrects already-clean regions.

    This estimator projects the desired high-pass correction onto a small set of
    period-locked Fourier harmonics *separately in each complete PWM cycle*.
    The coefficient median across cycles keeps only structure that recurs at the
    same phase; ordinary scene edges and object texture vary from cycle to cycle
    and are rejected by the median/MAD agreement test.  Coefficients are then
    smoothed only across X, preserving the global rolling-shutter timing while
    allowing the residual amplitude/color to change slowly across the scene.
    """
    if residual.ndim != 4 or support.ndim != 4 or support.shape[1] != 1:
        raise ValueError("cycle-consensus PWM inputs must be BCHW with scalar support")
    if residual.shape[0] != support.shape[0] or residual.shape[-2:] != support.shape[-2:]:
        raise ValueError("cycle-consensus PWM support must match residual")
    b, c, h, w = residual.shape
    if period <= 4.0 or h < int(max(16.0, 3.0 * period)):
        z = torch.zeros_like(residual)
        e = torch.zeros((b, 1, h, w), device=residual.device, dtype=residual.dtype)
        return z, e, e

    # Preserve only the band-scale residual before harmonic projection.
    base_sigma = max(12.0, float(period) * 1.25)
    target = _smooth_axis(residual.float(), base_sigma, "y") - residual.float()
    ww = support.float().clamp(0.0, 1.0)
    ref_target = None
    if reference_correction is not None:
        ref = reference_correction.float().to(device=residual.device)
        if ref.ndim != 4 or ref.shape[0] != b or ref.shape[1] != c:
            raise ValueError("PWM reference correction must match batch/channel count")
        if ref.shape[-2:] != (h, w):
            ref = F.interpolate(ref, size=(h, w), mode="bilinear", align_corners=False)
        ref_target = ref - _smooth_axis(ref, base_sigma, "y")

    # Keep the shortest modeled wavelength around 10-12 rows.  This is enough
    # for square-ish residual shoulders but deliberately avoids fitting fine
    # image texture.
    kmax = max(1, min(int(max_harmonics), int(max(1.0, math.floor(float(period) / 10.0)))))
    yy = torch.arange(h, device=residual.device, dtype=torch.float32)
    basis = []
    omega = 2.0 * math.pi / float(period)
    for k in range(1, kmax + 1):
        basis.append(torch.sin(yy * (omega * k)))
        basis.append(torch.cos(yy * (omega * k)))
    basis = torch.stack(basis, dim=0)  # JxH

    cycle_idx = torch.floor(yy / float(period)).long()
    coeff_cycles = []
    reference_cycles = []
    coverage_cycles = []
    for ci in range(int(cycle_idx.min()), int(cycle_idx.max()) + 1):
        rows = torch.nonzero(cycle_idx == ci, as_tuple=False).flatten()
        if rows.numel() < max(6, int(math.floor(0.70 * float(period)))):
            continue
        t = target[:, :, rows, :]
        rt = ref_target[:, :, rows, :] if ref_target is not None else None
        sw = ww[:, :, rows, :]
        wc = sw.expand(-1, c, -1, -1)
        # Remove a cycle-local DC term.  Broad scene level changes then cannot
        # masquerade as the fundamental merely because the finite image window
        # does not contain an integer number of periods.
        mean = (t * wc).sum(dim=-2, keepdim=True) / wc.sum(dim=-2, keepdim=True).clamp_min(1e-6)
        t = t - mean
        if rt is not None:
            rmean = (rt * wc).sum(dim=-2, keepdim=True) / wc.sum(dim=-2, keepdim=True).clamp_min(1e-6)
            rt = rt - rmean
        bb = basis[:, rows]  # JxR
        nums = []
        ref_nums = []
        dens = []
        for j in range(bb.shape[0]):
            q = bb[j].view(1, 1, -1, 1)
            nums.append((t * q * wc).sum(dim=-2))
            if rt is not None:
                ref_nums.append((rt * q * wc).sum(dim=-2))
            dens.append((q.square() * wc).sum(dim=-2))
        num = torch.stack(nums, dim=2)  # BxCxJxW
        den = torch.stack(dens, dim=2).clamp_min(1e-8)
        coeff_cycles.append(num / den)
        if rt is not None:
            reference_cycles.append(torch.stack(ref_nums, dim=2) / den)
        coverage_cycles.append(sw.mean(dim=-2))  # Bx1xW

    if len(coeff_cycles) < 4:
        z = torch.zeros_like(residual)
        e = torch.zeros((b, 1, h, w), device=residual.device, dtype=residual.dtype)
        return z, e, e

    stack = torch.stack(coeff_cycles, dim=0)  # NxBxCxJxW
    med = stack.median(dim=0).values
    mad = (stack - med.unsqueeze(0)).abs().median(dim=0).values
    sigma = 1.4826 * mad

    # Slow X regularization: the PWM timing/waveform is global, while spectral
    # response may vary between broad scene zones.  Do not let coefficients trace
    # individual objects.
    prototype_min_x = min(128.0, max(48.0, 0.064 * float(w)))
    sx = max(prototype_min_x, min(256.0, float(period) * max(0.75, float(sigma_x_ratio))))
    med = _smooth_axis(med, sx, "x")
    sigma = _smooth_axis(sigma, sx, "x")

    signal = torch.sqrt(med.square().sum(dim=2, keepdim=True))
    noise = torch.sqrt(sigma.square().sum(dim=2, keepdim=True)).clamp_min(2e-5)
    snr = signal / noise
    evidence_c = _smoothstep(snr, 0.055, 0.165)

    if reference_cycles:
        ref_stack = torch.stack(reference_cycles, dim=0)
        ref_med = ref_stack.median(dim=0).values
        ref_med = _smooth_axis(ref_med, sx, "x")
        ref_signal = torch.sqrt(ref_med.square().sum(dim=2, keepdim=True)).clamp_min(1e-6)
        dot = (med * ref_med).sum(dim=2, keepdim=True)
        align = dot / (signal * ref_signal).clamp_min(1e-8)
        ratio = signal / ref_signal
        align_gate = _smoothstep(align, 0.70, 0.95) * _smoothstep(ratio, 0.35, 0.90)
        ref_present = _smoothstep(ref_signal, 2e-4, 1.0e-3)
        reference_gate = (1.0 - ref_present) + ref_present * align_gate
        evidence_c = (evidence_c * reference_gate).clamp(0.0, 1.0)

    cov = torch.stack(coverage_cycles, dim=0).median(dim=0).values  # Bx1xW
    cov_gate = _smoothstep(cov, 0.04, 0.20).unsqueeze(2)  # Bx1x1xW
    evidence_c = (evidence_c * cov_gate).clamp(0.0, 1.0)

    # Reconstruct the correction at every row from the cycle-consensus harmonic
    # coefficients.  ``target`` was already defined with correction sign.
    field = torch.zeros_like(residual, dtype=torch.float32)
    for j in range(basis.shape[0]):
        q = basis[j].view(1, 1, h, 1)
        field = field + q * med[:, :, j:j+1, :]
    ev_channel = evidence_c.expand(-1, -1, h, -1)
    field = field * ev_channel

    ev_scalar = ev_channel.mean(dim=1, keepdim=True)
    return field.to(residual.dtype), ev_scalar.to(residual.dtype), ev_scalar.to(residual.dtype)



def _pwm_surface_cv_field(
    residual: torch.Tensor,
    support: torch.Tensor,
    *,
    period: float,
    max_harmonics: int = 5,
    reference_correction: torch.Tensor | None = None,
    surface_guide: torch.Tensor | None = None,
    analysis_width: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-validated, surface-conditioned PWM residual model.

    The v3 global cycle median is excellent on broad backgrounds but can paint a
    globally valid waveform onto a foreground surface whose residual response is
    different.  This v4 companion model is deliberately conservative:

    * each PWM cycle is held out while candidate harmonic models are estimated
      from the other cycles;
    * a local candidate may borrow from nearby X positions only when a slow
      Y/Cb/Cr guide says those samples belong to a similar surface;
    * both the broad global candidate and the local surface candidate must reduce
      held-out harmonic error versus doing nothing;
    * the neural correction is used only as a PWM-presence cue, not as a signed
      target, so the model may safely back away from local neural over-correction.

    Analysis is capped in X because PWM response varies slowly across columns.
    This keeps the extra validation path practical while preserving full vertical
    phase resolution.
    """
    if residual.ndim != 4 or support.ndim != 4 or support.shape[1] != 1:
        raise ValueError("surface-CV PWM inputs must be BCHW with scalar support")
    if residual.shape[0] != support.shape[0] or residual.shape[-2:] != support.shape[-2:]:
        raise ValueError("surface-CV PWM support must match residual")
    b, c, h, w = residual.shape
    if surface_guide is None or period <= 4.0 or h < int(max(16.0, 3.0 * period)):
        z = torch.zeros_like(residual)
        e = torch.zeros((b, 1, h, w), device=residual.device, dtype=residual.dtype)
        return z, e, e
    if surface_guide.ndim != 4 or surface_guide.shape[0] != b or surface_guide.shape[1] < 1:
        raise ValueError("PWM surface_guide must be BCHW with at least luma")

    aw = min(int(max(64, analysis_width)), w)
    scale_x = float(aw) / float(max(1, w))

    def _resize_x(t: torch.Tensor) -> torch.Tensor:
        if t.shape[-2:] != (h, w):
            t = F.interpolate(t.float(), size=(h, w), mode="bilinear", align_corners=False)
        if aw < w:
            return F.interpolate(t.float(), size=(h, aw), mode="bilinear", align_corners=False)
        return t.float()

    ra = _resize_x(residual)
    sa = _resize_x(support).clamp(0.0, 1.0)
    guide = _resize_x(surface_guide)
    refa = _resize_x(reference_correction.to(residual.device)) if reference_correction is not None else None

    base_sigma = max(12.0, float(period) * 1.25)
    target = _smooth_axis(ra, base_sigma, "y") - ra
    ref_target = None if refa is None else refa - _smooth_axis(refa, base_sigma, "y")

    # PWM-resistant scene guide.  Luma is converted to log space so surface
    # distance is closer to perceptual/exposure distance; chroma remains linear.
    gy = _smooth_axis(guide[:, 0:1], max(2.0, float(period) * 0.58), "y")
    if guide.shape[1] >= 3:
        gc = _smooth_axis(guide[:, 1:3], max(2.0, float(period) * 0.58), "y")
    else:
        gc = torch.zeros((b, 2, h, aw), device=residual.device, dtype=torch.float32)
    gmap = torch.cat([torch.log(gy.clamp_min(1e-4)), gc], dim=1)
    gmap = _smooth_axis(gmap, max(1.0, 1.5 * scale_x), "x")

    kmax = max(1, min(int(max_harmonics), int(max(1.0, math.floor(float(period) / 10.0)))))
    yy = torch.arange(h, device=residual.device, dtype=torch.float32)
    omega = 2.0 * math.pi / float(period)
    basis = []
    for k in range(1, kmax + 1):
        basis.append(torch.sin(yy * (omega * k)))
        basis.append(torch.cos(yy * (omega * k)))
    basis = torch.stack(basis, dim=0)

    cycle_idx = torch.floor(yy / float(period)).long()
    coeff_cycles: list[torch.Tensor] = []
    ref_cycles: list[torch.Tensor] = []
    coverage_cycles: list[torch.Tensor] = []
    guide_cycles: list[torch.Tensor] = []
    cycle_ids: list[int] = []
    for ci in range(int(cycle_idx.min()), int(cycle_idx.max()) + 1):
        rows = torch.nonzero(cycle_idx == ci, as_tuple=False).flatten()
        if rows.numel() < max(6, int(math.floor(0.70 * float(period)))):
            continue
        t = target[:, :, rows, :]
        sw = sa[:, :, rows, :]
        wc = sw.expand(-1, c, -1, -1)
        t = t - (t * wc).sum(dim=-2, keepdim=True) / wc.sum(dim=-2, keepdim=True).clamp_min(1e-6)
        rt = ref_target[:, :, rows, :] if ref_target is not None else None
        if rt is not None:
            rt = rt - (rt * wc).sum(dim=-2, keepdim=True) / wc.sum(dim=-2, keepdim=True).clamp_min(1e-6)

        bb = basis[:, rows]
        nums = []
        ref_nums = []
        dens = []
        for j in range(bb.shape[0]):
            q = bb[j].view(1, 1, -1, 1)
            nums.append((t * q * wc).sum(dim=-2))
            dens.append((q.square() * wc).sum(dim=-2))
            if rt is not None:
                ref_nums.append((rt * q * wc).sum(dim=-2))
        den = torch.stack(dens, dim=2).clamp_min(1e-8)
        coeff_cycles.append(torch.stack(nums, dim=2) / den)
        if rt is not None:
            ref_cycles.append(torch.stack(ref_nums, dim=2) / den)
        coverage_cycles.append(sw.mean(dim=-2))
        guide_cycles.append(gmap[:, :, rows, :].mean(dim=-2))
        cycle_ids.append(ci)

    if len(coeff_cycles) < 5:
        z = torch.zeros_like(residual)
        e = torch.zeros((b, 1, h, w), device=residual.device, dtype=residual.dtype)
        return z, e, e

    stack = torch.stack(coeff_cycles, dim=0)       # NxBxCxJxW
    coverage = torch.stack(coverage_cycles, dim=0) # NxBx1xW
    guides = torch.stack(guide_cycles, dim=0)      # NxBx3xW
    ref_stack = torch.stack(ref_cycles, dim=0) if ref_cycles else None
    n_cycles = stack.shape[0]
    inds = torch.arange(n_cycles, device=residual.device, dtype=torch.float32)
    sx_global = max(6.0, 128.0 * scale_x)

    # Pre-shift neighboring-X samples once.  The local held-out model can then
    # find the same physical surface even when a sloped shoulder/arm means that
    # surface does not occupy exactly the same X coordinate in another cycle.
    dxs = (-24, -12, 0, 12, 24)

    def _shift_x(t: torch.Tensor, dx: int) -> torch.Tensor:
        if dx == 0:
            return t
        if dx > 0:
            return torch.cat([t[..., dx:], t[..., -1:].expand(*t.shape[:-1], dx)], dim=-1)
        d = -dx
        return torch.cat([t[..., :1].expand(*t.shape[:-1], d), t[..., :-d]], dim=-1)

    shifted_stack = [_shift_x(stack, d) for d in dxs]
    shifted_guides = [_shift_x(guides, d) for d in dxs]
    shifted_coverage = [_shift_x(coverage, d) for d in dxs]

    chosen_coeffs = []
    chosen_evidence = []
    for i in range(n_cycles):
        # Broad global candidate, estimated without the target cycle.
        held_out = torch.cat([stack[:i], stack[i + 1:]], dim=0)
        global_model = held_out.median(dim=0).values
        global_model = _smooth_axis(global_model, sx_global, "x")

        observed = stack[i]
        observed_power = observed.square().sum(dim=2, keepdim=True).clamp_min(1e-10)
        g_power = global_model.square().sum(dim=2, keepdim=True).clamp_min(1e-8)
        g_dot = (observed * global_model).sum(dim=2, keepdim=True)
        g_align = g_dot / torch.sqrt(g_power * observed_power).clamp_min(1e-8)
        g_gain = (g_dot / g_power).clamp(0.0, 1.25)
        global_fit = global_model * g_gain
        zero_error = observed_power
        global_error = (observed - global_fit).square().sum(dim=2, keepdim=True)
        global_benefit = (zero_error - global_error) / zero_error.clamp_min(1e-8)
        global_ev = (
            _smoothstep(g_align, 0.30, 0.82)
            * _smoothstep(global_benefit, 0.01, 0.18)
            * _smoothstep(torch.sqrt(g_power), 1.0e-4, 7.0e-4)
        )

        if ref_stack is not None:
            held_ref = torch.cat([ref_stack[:i], ref_stack[i + 1:]], dim=0).median(dim=0).values
            held_ref = _smooth_axis(held_ref, sx_global, "x")
            ref_signal = torch.sqrt(held_ref.square().sum(dim=2, keepdim=True))
            # Presence only: local neural over-correction is allowed to be
            # corrected in the opposite direction by the held-out residual fit.
            global_ev = global_ev * (0.35 + 0.65 * _smoothstep(ref_signal, 8.0e-5, 7.0e-4))

        # 2-D same-surface candidate, also excluding the target cycle entirely.
        cycle_weight = torch.exp(-0.5 * ((inds - float(i)) / 3.2).square()).view(n_cycles, 1, 1, 1)
        cycle_weight[i] = 0.0
        local_num = torch.zeros_like(global_model)
        local_den = torch.zeros((b, 1, aw), device=residual.device, dtype=torch.float32)
        for dx, ss, gg, ccov in zip(dxs, shifted_stack, shifted_guides, shifted_coverage):
            dg = gg - guides[i:i + 1]
            lum_dist = (dg[:, :, 0:1, :].abs() / 0.20).square()
            chroma_dist = (dg[:, :, 1:3, :].square().sum(dim=2, keepdim=True).sqrt() / 0.060).square()
            x_weight = math.exp(-0.5 * (float(dx) / 22.0) ** 2)
            ww = (
                cycle_weight
                * float(x_weight)
                * torch.exp(-0.5 * (lum_dist + chroma_dist))
                * ccov.clamp(0.03, 1.0)
            )
            local_num = local_num + (ss * ww.unsqueeze(2)).sum(dim=0)
            local_den = local_den + ww.sum(dim=0)
        local_model = local_num / local_den.unsqueeze(2).clamp_min(1e-6)
        local_model = _smooth_axis(local_model, max(1.5, 10.0 * scale_x), "x")

        l_power = local_model.square().sum(dim=2, keepdim=True).clamp_min(1e-8)
        l_dot = (observed * local_model).sum(dim=2, keepdim=True)
        l_align = l_dot / torch.sqrt(l_power * observed_power).clamp_min(1e-8)
        l_gain = (l_dot / l_power).clamp(0.0, 1.35)
        local_fit = local_model * l_gain
        local_error = (observed - local_fit).square().sum(dim=2, keepdim=True)
        local_benefit = (zero_error - local_error) / zero_error.clamp_min(1e-8)
        local_ev = (
            _smoothstep(l_align, 0.28, 0.78)
            * _smoothstep(local_benefit, 0.01, 0.15)
            * _smoothstep(torch.sqrt(l_power), 8.0e-5, 6.0e-4)
            * _smoothstep(local_den.unsqueeze(2), 0.35, 1.50)
        )

        # Select local only when it improves held-out prediction over the global
        # candidate.  If neither candidate predicts the target cycle, both fade
        # toward zero instead of inventing a foreground PWM waveform.
        local_advantage = (global_error - local_error) / zero_error.clamp_min(1e-8)
        choose_local = local_ev * _smoothstep(local_advantage, 0.01, 0.12)
        chosen = (1.0 - choose_local) * (global_fit * global_ev) + choose_local * (local_fit * local_ev)
        ev = (1.0 - choose_local) * global_ev + choose_local * local_ev
        chosen_coeffs.append(chosen)
        chosen_evidence.append(ev.clamp(0.0, 1.0))

    coeff = torch.stack(chosen_coeffs, dim=0)
    evidence = torch.stack(chosen_evidence, dim=0)
    centers = torch.tensor([(ci + 0.5) * float(period) for ci in cycle_ids], device=residual.device)
    nearest = torch.argmin(
        (torch.arange(h, device=residual.device, dtype=torch.float32).view(-1, 1) - centers.view(1, -1)).abs(),
        dim=1,
    )
    coeff_y = torch.empty((b, c, basis.shape[0], h, aw), device=residual.device, dtype=torch.float32)
    evidence_y = torch.empty((b, c, 1, h, aw), device=residual.device, dtype=torch.float32)
    for i in range(n_cycles):
        rows = nearest == i
        coeff_y[:, :, :, rows, :] = coeff[i].unsqueeze(3)
        evidence_y[:, :, :, rows, :] = evidence[i].unsqueeze(3)

    coeff_y = _smooth_axis(
        coeff_y.reshape(b, c * basis.shape[0], h, aw), max(4.0, float(period) * 0.18), "y"
    ).reshape(b, c, basis.shape[0], h, aw)
    ev_map = _smooth_axis(
        evidence_y.reshape(b, c, h, aw), max(3.0, float(period) * 0.14), "y"
    ).clamp(0.0, 1.0)

    field = torch.zeros((b, c, h, aw), device=residual.device, dtype=torch.float32)
    for j in range(basis.shape[0]):
        field = field + basis[j].view(1, 1, h, 1) * coeff_y[:, :, j]
    field = field * ev_map

    if aw < w:
        field = F.interpolate(field, size=(h, w), mode="bilinear", align_corners=False)
        ev_map = F.interpolate(ev_map, size=(h, w), mode="bilinear", align_corners=False)
    ev_scalar = ev_map.mean(dim=1, keepdim=True)
    return field.to(residual.dtype), ev_scalar.to(residual.dtype), ev_scalar.to(residual.dtype)


def _pwm_xonly_amplitude(
    residual: torch.Tensor,
    correction: torch.Tensor,
    support: torch.Tensor,
    edge_support: torch.Tensor,
    *,
    period: float,
    sigma_x_ratio: float,
    corr_low: float,
    corr_high: float,
    max_gain: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit PWM amplitude only across X, never independently along Y.

    A validated PWM waveform has one global phase/duty cycle.  Allowing the
    ordinary 2-D adaptive fit to vary along Y can make the gain collapse at a
    sharp state transition and creates patch-shaped or band-shaped artifacts.
    Here every processing column gets only one scalar amplitude (one shared
    scalar for the CbCr vector), broadly regularized across X.
    """
    if correction.shape[-1] == 1:
        correction = correction.expand(-1, correction.shape[1], -1, residual.shape[-1])
    if correction.shape != residual.shape:
        raise ValueError("PWM X-only correction must match residual")

    w = (support.float().clamp(0.0, 1.0) * edge_support.float().clamp(0.0, 1.0)).clamp(0.0, 1.0)
    wc = w.expand(-1, residual.shape[1], -1, -1)

    # Remove only a slow per-column baseline.  Do not Gaussian-bandpass at the
    # ordinary profile scale: the higher harmonics are the desired square edge.
    base_sigma = max(12.0, float(period) * 1.25)
    target = _smooth_axis(residual.float(), base_sigma, "y") - residual.float()
    corr = correction.float()

    num = (target * corr * wc).sum(dim=1, keepdim=True).sum(dim=-2, keepdim=True)
    den = (corr.square() * wc).sum(dim=1, keepdim=True).sum(dim=-2, keepdim=True)
    power = (target.square() * wc).sum(dim=1, keepdim=True).sum(dim=-2, keepdim=True)
    coverage = w.mean(dim=-2, keepdim=True)

    # The amplitude may differ between broad scene zones, but must not trace
    # individual objects.  Give it roughly one whole PWM period of X support.
    sx = max(48.0, min(256.0, float(period) * max(0.75, float(sigma_x_ratio))))
    num = _smooth_axis(num, sx, "x")
    den = _smooth_axis(den, sx, "x")
    power = _smooth_axis(power, sx, "x")
    coverage = _smooth_axis(coverage, sx, "x")

    fit = num / den.clamp_min(1e-8)
    signed_corr = num / torch.sqrt((den * power).clamp_min(1e-12))
    evidence = _smoothstep(signed_corr, float(corr_low), float(corr_high))
    evidence = evidence * _smoothstep(coverage, 0.03, 0.18)
    amp_x = fit.clamp(0.0, float(max_gain)) * evidence
    amp_x = amp_x.clamp(0.0, float(max_gain))
    amp = amp_x.expand(-1, 1, residual.shape[-2], -1)
    ev = signed_corr.clamp(0.0, 1.0).expand(-1, 1, residual.shape[-2], -1)
    return amp, ev


def _pwm_step_profile_correction(
    row_y: torch.Tensor,
    row_c: torch.Tensor,
    row_weight: torch.Tensor,
    *,
    period: float,
    transition_ratio: float = 0.010,
    min_duty: float = 0.08,
    max_duty: float = 0.92,
    min_transition_score: float = 2.0,
    transition_phases: tuple[float, float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Estimate a two-state PWM/step correction from robust row profiles.

    The row signal is folded by the detected period, so a real repeating LED
    transition reinforces while isolated scene edges land at unrelated phases.
    Two globally coherent transition phases are then selected and each plateau
    receives one robust Y/CbCr level.  Only the *correction field* is piecewise
    constant; source image pixels are never spatially filtered.

    Returns ``(corr_log_y, corr_c, confidence_rows)`` in BxCxHx1 form, or None
    when the folded profile does not contain a trustworthy two-transition PWM
    waveform.  The caller then falls back to the ordinary smooth profile mode.
    """
    if period <= 2.0:
        return None
    if row_y.ndim != 4 or row_c.ndim != 4 or row_weight.ndim != 4:
        raise ValueError("PWM profile inputs must be BCHW row tensors")
    b, _, h, _ = row_y.shape
    if h < max(12, int(round(period * 1.5))):
        return None
    if not (0.0 < min_duty < max_duty < 1.0):
        raise ValueError("PWM duty limits must satisfy 0 < min < max < 1")

    # About one phase bin per source row of the detected period preserves sharp
    # edges while capping memory/work on extremely broad periods.
    bins = max(16, min(768, int(round(float(period)))))
    yy = torch.arange(h, device=row_y.device, dtype=torch.float32)
    phase = torch.remainder(yy / float(period), 1.0)
    phase_idx = torch.floor(phase * bins).long().clamp(0, bins - 1)

    corr_y_batches: list[torch.Tensor] = []
    corr_c_batches: list[torch.Tensor] = []
    conf_batches: list[torch.Tensor] = []
    any_valid = False

    for bi in range(b):
        wrow = row_weight[bi, 0, :, 0].float().clamp(0.0, 1.0)
        den = torch.zeros(bins, device=row_y.device, dtype=torch.float32)
        den.scatter_add_(0, phase_idx, wrow)
        if int((den > 1e-4).sum()) < max(8, bins // 3):
            corr_y_batches.append(torch.zeros((1, h, 1), device=row_y.device, dtype=torch.float32))
            corr_c_batches.append(torch.zeros((2, h, 1), device=row_y.device, dtype=torch.float32))
            conf_batches.append(torch.zeros((1, h, 1), device=row_y.device, dtype=torch.float32))
            continue

        folded_channels = []
        source = torch.cat((row_y[bi, :, :, 0], row_c[bi, :, :, 0]), dim=0).float()  # 3xH
        for ci in range(3):
            phase_values = []
            for pi in range(bins):
                keep = (phase_idx == pi) & (wrow > 0.05)
                vv = source[ci, keep]
                if vv.numel():
                    phase_values.append(vv.median())
                else:
                    phase_values.append(torch.tensor(0.0, device=row_y.device, dtype=torch.float32))
            folded_channels.append(torch.stack(phase_values))
        folded = torch.stack(folded_channels, dim=0)  # 3xN

        # Fill the rare unsupported phase bin by circular normalized smoothing.
        valid = (den > 1e-4).float()
        if bool((valid < 0.5).any()):
            radius = max(1, min(8, bins // 8))
            kernel = torch.ones(1, 1, 2 * radius + 1, device=folded.device, dtype=folded.dtype)
            vp = F.pad(valid.view(1, 1, -1), (radius, radius), mode="circular")
            dd = F.conv1d(vp, kernel).view(-1)
            for ci in range(3):
                fp = F.pad((folded[ci] * valid).view(1, 1, -1), (radius, radius), mode="circular")
                ff = F.conv1d(fp, kernel).view(-1) / dd.clamp_min(1e-6)
                folded[ci] = torch.where(valid > 0.5, folded[ci], ff)

        # Remove only the folded DC.  The input row profiles have already had a
        # linear scene trend removed; retaining the plateau jump is the point of
        # this mode.
        phase_w = den / den.sum().clamp_min(1e-6)
        folded = folded - (folded * phase_w.unsqueeze(0)).sum(dim=-1, keepdim=True)
        tiny_sigma = max(0.35, min(2.0, float(transition_ratio) * bins * 0.45))
        sharp = _circular_smooth_profile(folded, tiny_sigma)

        # Detect shared transitions using scale-normalized Y plus vector chroma.
        med = sharp.median(dim=-1, keepdim=True).values
        mad = (sharp - med).abs().median(dim=-1, keepdim=True).values.clamp_min(1e-5)
        scale = (1.4826 * mad).clamp_min(1e-5)
        # Prevent a nearly-neutral channel with microscopic MAD from dominating
        # transition detection.  Chroma remains useful evidence, but its noise
        # floor is tied loosely to the luminance residual scale.
        chroma_floor = (0.15 * scale[0:1]).clamp_min(2e-4)
        scale[1:3] = torch.maximum(scale[1:3], chroma_floor.expand_as(scale[1:3]))
        deriv = torch.roll(sharp, shifts=-1, dims=-1) - sharp
        zderiv = deriv / scale
        grad = torch.sqrt(
            zderiv[0].square()
            + 0.35 * zderiv[1].square()
            + 0.35 * zderiv[2].square()
        )
        gmed = grad.median()
        gmad = (grad - gmed).abs().median().clamp_min(1e-6)
        noise = (gmed + 1.4826 * gmad).clamp_min(1e-6)
        neural_phase_conf = 0.0
        if transition_phases is not None:
            # Neural correction locks the physical edge pair; refine each edge
            # only a few phase bins on the *residual* profile.  This avoids a
            # dominant scene boundary choosing a completely different duty cycle.
            ph1, ph2, neural_phase_conf = transition_phases
            e1 = int(round(float(ph1 % 1.0) * bins)) % bins
            e2 = int(round(float(ph2 % 1.0) * bins)) % bins
            radius = max(2, min(8, int(round(0.08 * bins))))
            cand1 = sorted({(e1 + d) % bins for d in range(-radius, radius + 1)})
            cand2 = sorted({(e2 + d) % bins for d in range(-radius, radius + 1)})
            best_pair: tuple[float, int, int, float] | None = None
            for a in cand1:
                va = zderiv[:, a]
                van = torch.sqrt(va.square().sum()).clamp_min(1e-8)
                for b2 in cand2:
                    dist_ab = (b2 - a) % bins
                    if dist_ab < int(math.ceil(min_duty * bins)) or dist_ab > int(math.floor(max_duty * bins)):
                        continue
                    vb = zderiv[:, b2]
                    vbn = torch.sqrt(vb.square().sum()).clamp_min(1e-8)
                    opp = float((-torch.dot(va, vb) / (van * vbn)).clamp(0.0, 1.0))
                    # Residual edges may already be weak after Restormer; the
                    # neural phase prior is allowed to carry part of the score.
                    pair = float(torch.minimum(grad[a], grad[b2])) * (0.35 + 0.65 * opp)
                    if best_pair is None or pair > best_pair[0]:
                        best_pair = (pair, a, b2, opp)
            if best_pair is None:
                p1 = e1
                p2 = e2
                opposition = 0.0
            else:
                _, p1, p2, opposition = best_pair
            peak_score = float(torch.minimum(grad[p1], grad[p2]) / noise) * max(0.35, opposition)
            # A trusted neural edge pair does not require the residual derivative
            # itself to clear the full legacy transition threshold. The later
            # plateau contrast, X-only fit and no-harm checks still have vetoes.
            min_score_here = max(0.45, float(min_transition_score) * (1.0 - 0.70 * float(neural_phase_conf)))
            if peak_score < min_score_here:
                # Keep the neural phases exactly rather than jumping to an
                # unrelated scene edge; weak residuals can legitimately have
                # little derivative after the neural pass.
                p1, p2 = e1, e2
                peak_score = min_score_here
        else:
            p1 = int(torch.argmax(grad).item())
            dist = torch.remainder(torch.arange(bins, device=grad.device) - p1, bins)
            allowed = (dist >= int(math.ceil(min_duty * bins))) & (dist <= int(math.floor(max_duty * bins)))
            if not bool(allowed.any()):
                corr_y_batches.append(torch.zeros((1, h, 1), device=row_y.device, dtype=torch.float32))
                corr_c_batches.append(torch.zeros((2, h, 1), device=row_y.device, dtype=torch.float32))
                conf_batches.append(torch.zeros((1, h, 1), device=row_y.device, dtype=torch.float32))
                continue
            # The second edge of a two-state PWM cycle must reverse the first
            # Y/CbCr transition vector.  Magnitude-only selection can lock onto two
            # same-polarity scene/noise edges and create false plateaus.
            v1 = zderiv[:, p1]
            v1n = torch.sqrt((v1.square()).sum()).clamp_min(1e-8)
            opposite = torch.zeros_like(grad)
            for pi in range(bins):
                if not bool(allowed[pi]):
                    continue
                v2 = zderiv[:, pi]
                v2n = torch.sqrt((v2.square()).sum()).clamp_min(1e-8)
                opposite[pi] = (-torch.dot(v1, v2) / (v1n * v2n)).clamp(0.0, 1.0)
            pair_score = grad * opposite
            masked_grad = torch.where(allowed, pair_score, torch.full_like(pair_score, -1.0))
            p2 = int(torch.argmax(masked_grad).item())
            opposition = float(opposite[p2])
            peak_score = float(torch.minimum(grad[p1], grad[p2]) / noise) * opposition
            if opposition < 0.35 or peak_score < float(min_transition_score):
                corr_y_batches.append(torch.zeros((1, h, 1), device=row_y.device, dtype=torch.float32))
                corr_c_batches.append(torch.zeros((2, h, 1), device=row_y.device, dtype=torch.float32))
                conf_batches.append(torch.zeros((1, h, 1), device=row_y.device, dtype=torch.float32))
                continue

        lo, hi = sorted((p1, p2))
        idx = torch.arange(bins, device=folded.device)
        arc_a = (idx > lo) & (idx <= hi)
        arc_b = ~arc_a
        # Ignore a tiny zone immediately beside transitions while estimating the
        # plateau levels. This is robust to finite LED rise/fall time and JPEG
        # ringing without softening the final step location.
        margin = max(1, min(3, int(round(float(transition_ratio) * bins * 1.5))))
        for pp in (p1, p2):
            cd = torch.minimum(torch.remainder(idx - pp, bins), torch.remainder(pp - idx, bins))
            keep = cd > margin
            arc_a &= keep
            arc_b &= keep
        if int(arc_a.sum()) < 2 or int(arc_b.sum()) < 2:
            corr_y_batches.append(torch.zeros((1, h, 1), device=row_y.device, dtype=torch.float32))
            corr_c_batches.append(torch.zeros((2, h, 1), device=row_y.device, dtype=torch.float32))
            conf_batches.append(torch.zeros((1, h, 1), device=row_y.device, dtype=torch.float32))
            continue

        level_a = sharp[:, arc_a].median(dim=-1).values
        level_b = sharp[:, arc_b].median(dim=-1).values
        contrast = torch.sqrt(
            ((level_a[0] - level_b[0]) / scale[0, 0]).square()
            + 0.45 * ((level_a[1] - level_b[1]) / scale[1, 0]).square()
            + 0.45 * ((level_a[2] - level_b[2]) / scale[2, 0]).square()
        )
        if float(contrast) < 0.55:
            corr_y_batches.append(torch.zeros((1, h, 1), device=row_y.device, dtype=torch.float32))
            corr_c_batches.append(torch.zeros((2, h, 1), device=row_y.device, dtype=torch.float32))
            conf_batches.append(torch.zeros((1, h, 1), device=row_y.device, dtype=torch.float32))
            continue

        step = torch.empty_like(sharp)
        raw_arc_a = (idx > lo) & (idx <= hi)
        step[:, raw_arc_a] = level_a[:, None]
        step[:, ~raw_arc_a] = level_b[:, None]
        step = step - (step * phase_w.unsqueeze(0)).sum(dim=-1, keepdim=True)
        feather_sigma = max(0.30, min(2.25, float(transition_ratio) * bins))
        step = _circular_smooth_profile(step, feather_sigma)

        # ``step`` estimates the residual illumination state, therefore the
        # desired correction is its negative.
        corr_phase = -step
        row_phase = corr_phase[:, phase_idx]  # 3xH
        if transition_phases is not None:
            confidence = max(0.35 * float(neural_phase_conf), min(1.0, float(neural_phase_conf)))
        else:
            confidence = max(0.0, min(1.0, (peak_score - float(min_transition_score)) / 3.0))
        confidence *= max(0.0, min(1.0, float(contrast) / 2.0))
        any_valid = any_valid or confidence > 0.0
        corr_y_batches.append(row_phase[0:1].unsqueeze(-1))
        corr_c_batches.append(row_phase[1:3].unsqueeze(-1))
        conf_batches.append(torch.full((1, h, 1), confidence, device=row_y.device, dtype=torch.float32))

    if not any_valid:
        return None
    return (
        torch.stack(corr_y_batches, dim=0),
        torch.stack(corr_c_batches, dim=0),
        torch.stack(conf_batches, dim=0),
    )


def _residual_row_profile_correction(
    y: torch.Tensor,
    c: torch.Tensor,
    *,
    support: torch.Tensor,
    application_gate: torch.Tensor,
    edge_support: torch.Tensor,
    user_mask: torch.Tensor | None = None,
    high_strength_guard: torch.Tensor | None = None,
    period: float,
    period_confidence: float,
    luma_strength: float,
    chroma_strength: float,
    profile_mode: str = "smooth",
    narrow_ratio: float = 0.035,
    pwm_transition_ratio: float = 0.010,
    pwm_min_duty: float = 0.08,
    pwm_max_duty: float = 0.92,
    pwm_min_transition_score: float = 2.0,
    pwm_transition_phases: tuple[float, float, float] | None = None,
    pwm_refine_auto_timing: bool = True,
    pwm_reference_luma: torch.Tensor | None = None,
    pwm_reference_chroma: torch.Tensor | None = None,
    pwm_periods: tuple[float, ...] | None = None,
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

    profile_mode = str(profile_mode).lower()
    if profile_mode not in {"smooth", "pwm"}:
        raise ValueError("profile_mode must be 'smooth' or 'pwm'")

    # PWM needs broad row evidence, including the actual state transitions.
    # The ordinary profile support deliberately rejects scene/image edges and
    # strong tonal extremes; on strobed concert frames that can also reject the
    # LED transition rows themselves.  For PWM estimation use only a slow SNR
    # gate. Robust cross-column consensus + repeated opposite edge phases provide
    # the scene-edge rejection instead. Smooth mode keeps the historical support.
    analysis_support = support
    if profile_mode == "pwm":
        pwm_level = _smooth_axis(y.float(), max(2.0, float(period) * 0.40), "y")
        analysis_support = _smoothstep(pwm_level, 0.006, 0.035).clamp(0.0, 1.0)

    logy = torch.log(y.float().clamp_min(0.0) + float(eps))
    logy_res = _remove_linear_y_trend(logy, analysis_support)
    c_res = _remove_linear_y_trend(c.float(), analysis_support)

    row_y, valid_y, conf_y, _ = _robust_row_consensus(
        logy_res, analysis_support, huber_k=huber_k, min_coverage=min_coverage
    )
    row_c, valid_c, conf_c, _ = _robust_row_consensus(
        c_res, analysis_support, huber_k=huber_k, min_coverage=min_coverage
    )
    fill_sigma = max(1.0, period * 0.06)
    row_y = _fill_row_profile(row_y, valid_y, fill_sigma)
    row_c = _fill_row_profile(row_c, valid_c, fill_sigma)

    # Smooth mode keeps the established Gaussian band-pass behavior.  PWM mode
    # phase-folds the robust row consensus and estimates two piecewise-constant
    # illumination states with sharp shared Y/CbCr transitions.  If the folded
    # profile does not contain a convincing repeating two-state waveform, fall
    # back to the proven smooth estimator rather than forcing scene edges into
    # a step correction.
    pwm_fit = None
    pwm_template = None
    working_narrow_ratio = float(narrow_ratio)
    working_base_ratio = float(base_ratio)
    if profile_mode == "pwm":
        if pwm_transition_phases is not None:
            # Keep the measured period and edge phases exact.  Earlier builds
            # applied a fixed 0.989 period shrink plus a 2.5%-of-period edge
            # inset that happened to help one prototype image.  On fine PWM
            # bands that bias accumulates phase error over many cycles and can
            # make the global X-only least-squares fit collapse toward zero.
            # Timing is now refined by the independent residual edge cue before
            # reaching this stage, so no image-specific calibration is applied.
            period = float(period)
            pwm_transition_phases = (
                float(pwm_transition_phases[0]) % 1.0,
                float(pwm_transition_phases[1]) % 1.0,
                float(pwm_transition_phases[2]),
            )
            # Timing supplies geometry only; amplitudes are fitted independently
            # from the post-neural residual.
            pwm_template = _pwm_unit_step_template(
                batch=y.shape[0], height=y.shape[-2], period=period,
                transition_phases=pwm_transition_phases,
                transition_ratio=pwm_transition_ratio,
                device=y.device, dtype=y.dtype,
            )
            corr_log_y = pwm_template
            corr_c = pwm_template.expand(-1, 2, -1, -1)
            pwm_confidence_rows = torch.full_like(
                conf_y, max(0.0, min(1.0, float(pwm_transition_phases[2])))
            )
        else:
            # Fallback when the neural correction does not expose a trustworthy
            # edge train: retain the residual-only phase-folded two-state fit.
            row_weight = torch.maximum(conf_y, 0.55 * conf_c) * torch.maximum(valid_y, valid_c)
            pwm_fit = _pwm_step_profile_correction(
                row_y, row_c, row_weight,
                period=period,
                transition_ratio=pwm_transition_ratio,
                min_duty=pwm_min_duty,
                max_duty=pwm_max_duty,
                min_transition_score=pwm_min_transition_score,
                transition_phases=None,
            )
        working_narrow_ratio = min(float(narrow_ratio), max(0.006, float(pwm_transition_ratio) * 1.25))
        working_base_ratio = max(float(base_ratio), 0.35)

    if pwm_template is not None:
        pass
    elif pwm_fit is not None:
        corr_log_y, corr_c, pwm_confidence_rows = pwm_fit
    else:
        narrow_sigma = max(0.75, period * float(narrow_ratio))
        base_sigma = max(narrow_sigma + 0.5, period * float(base_ratio))
        narrow_y = _smooth_axis(row_y, narrow_sigma, "y")
        base_y = _smooth_axis(row_y, base_sigma, "y")
        narrow_c = _smooth_axis(row_c, narrow_sigma, "y")
        base_c = _smooth_axis(row_c, base_sigma, "y")
        corr_log_y = base_y - narrow_y
        corr_c = base_c - narrow_c
        pwm_confidence_rows = torch.ones_like(conf_y)

    period_weight = 0.85 + 0.15 * float(max(0.0, min(1.0, period_confidence)))
    locked_periods = tuple(float(p) for p in (pwm_periods or (period,)) if float(p) > 2.0)
    # A multi-LED mixture does not necessarily fold to one clean two-transition
    # square wave.  When cumulative neural timing exists, cycle-consensus and the
    # v5 multi-period basis can operate directly from the detected period family
    # even if the single-step template fit is ambiguous.
    pwm_advanced = (
        profile_mode == "pwm" and bool(locked_periods)
        and (pwm_template is not None or pwm_reference_luma is not None or pwm_reference_chroma is not None)
    )
    pwm_active = profile_mode == "pwm" and (pwm_advanced or pwm_fit is not None)

    if pwm_active:
        # Second-generation PWM path: once a repeated edge-locked waveform is
        # accepted, keep its phase/duty global and let only one scalar amplitude
        # vary across X.  Do not reuse the normal scene-shaped application gate;
        # it was designed for smooth profiles and can carve people/fixtures into
        # a sharp square-wave correction.
        if user_mask is None:
            user_gate = torch.ones_like(y)
        else:
            user_gate = user_mask.float().clamp(0.0, 1.0)
            if user_gate.ndim != 4 or user_gate.shape[1] != 1 or user_gate.shape[-2:] != y.shape[-2:]:
                raise ValueError("PWM user_mask must be Bx1xHxW and match Y")

        # Use a band-resistant signal level only as an SNR guard.  It varies
        # slowly across the PWM states, so a dark state does not receive less
        # correction merely because the flicker itself made that row darker.
        signal_level = _smooth_axis(y.float(), max(2.0, float(period) * 0.40), "y")
        signal_level = _smooth_axis(signal_level, 2.0, "x")
        reliability_y = _smoothstep(signal_level, 0.030, 0.095)
        reliability_c = _smoothstep(signal_level, 0.040, 0.120)

        # A successful phase fold has already enforced repeated transitions and
        # opposite edge polarity.  Confidence is exposed in diagnostics, while
        # actual authority is determined by the X-only fit + no-harm validator.
        row_conf_y = (pwm_confidence_rows * period_weight).clamp(0.0, 1.0)
        row_conf_c = (pwm_confidence_rows * period_weight).clamp(0.0, 1.0)
        apply_y = (user_gate * reliability_y * period_weight).clamp(0.0, 1.0)
        apply_c_scalar = (user_gate * reliability_c * period_weight).clamp(0.0, 1.0)
        apply_c = apply_c_scalar.expand(-1, 2, -1, -1)

        corr_y_map = corr_log_y.expand(-1, 1, -1, y.shape[-1])
        corr_c_map = corr_c.expand(-1, 2, -1, c.shape[-1])

        if pwm_advanced:
            # Independent edge/spectral timing has established one or more PWM
            # period.  Keep the proven v3/v4 result as a conservative baseline,
            # then let v5 refine that field only inside radiometrically coherent
            # regions that pass held-out X validation.  Additional non-harmonic
            # PWM periods discovered in local X strips are modeled independently;
            # same-period/different-phase lamps are handled by the segment's local
            # sine/cosine coefficients.
            active_periods = locked_periods if locked_periods else (float(period),)

            # v6 escape hatch for the failure mode where Restormer contributes
            # almost no PWM correction but the visible image contains one very
            # strong, globally phase-locked source.  v5's per-segment sine/cosine
            # freedom can overfit scene structure in this case.  If the same
            # phase is independently present across most perpendicular scene
            # strips, lock phase globally and fit only local positive amplitude.
            phase_lock_used = False
            phase_lock_y = phase_lock_c = phase_lock_amp_y = phase_lock_amp_c = None
            phase_lock_ev_y = phase_lock_ev_c = None
            if len(active_periods) == 1:
                pl_coh, pl_amp, pl_votes, pl_ref_ratio = _pwm_phase_lock_diagnostics(
                    logy_res, analysis_support, period=float(active_periods[0]),
                    reference_correction=pwm_reference_luma,
                )
                phase_lock_used = bool(
                    pl_coh >= 0.88 and pl_votes >= 3 and pl_amp >= 0.010
                    and pl_ref_ratio <= 0.35
                )
                if phase_lock_used:
                    phase_lock_y, phase_lock_amp_y, phase_lock_ev_y = _pwm_phase_locked_local_field(
                        logy_res, analysis_support, period=float(active_periods[0]), max_gain=1.35,
                        surface_guide=torch.cat([y.float(), c.float()], dim=1),
                    )
                    phase_lock_c, phase_lock_amp_c, phase_lock_ev_c = _pwm_phase_locked_local_field(
                        c_res, analysis_support, period=float(active_periods[0]), max_gain=1.35,
                        surface_guide=torch.cat([y.float(), c.float()], dim=1),
                    )

            global_y, global_amp_y, global_ev_y = _pwm_cycle_consensus_field(
                logy_res, analysis_support, period=float(active_periods[0]),
                sigma_x_ratio=adaptive_x_ratio, max_harmonics=5,
                reference_correction=pwm_reference_luma,
            )
            global_c, global_amp_c, global_ev_c = _pwm_cycle_consensus_field(
                c_res, analysis_support, period=float(active_periods[0]),
                sigma_x_ratio=adaptive_x_ratio, max_harmonics=5,
                reference_correction=pwm_reference_chroma,
            )

            # Preserve the v4 held-out surface model for the primary period as
            # the baseline.  This means v5 cannot regress a frame merely because
            # its segmentation is unhelpful: zero segmented authority reproduces
            # the established v4 field exactly.
            baseline_y = global_y
            baseline_c = global_c
            fallback_y = global_y
            fallback_c = global_c
            baseline_amp_y = global_amp_y
            baseline_amp_c = global_amp_c
            baseline_ev_y = global_ev_y
            baseline_ev_c = global_ev_c
            surface_guide = torch.cat([y.float(), c.float()], dim=1)
            if adaptive and not phase_lock_used:
                surface_y, surface_amp_y, surface_ev_y = _pwm_surface_cv_field(
                    logy_res, analysis_support, period=float(active_periods[0]), max_harmonics=5,
                    reference_correction=pwm_reference_luma, surface_guide=surface_guide,
                )
                surface_c, surface_amp_c, surface_ev_c = _pwm_surface_cv_field(
                    c_res, analysis_support, period=float(active_periods[0]), max_harmonics=5,
                    reference_correction=pwm_reference_chroma, surface_guide=surface_guide,
                )
                surface_authority = _smoothstep(signal_level, 0.16, 0.32)
                cv_any = torch.maximum(
                    surface_ev_y.amax(dim=(-2, -1), keepdim=True),
                    surface_ev_c.amax(dim=(-2, -1), keepdim=True),
                )
                surface_authority = surface_authority * _smoothstep(cv_any, 1e-4, 1e-2)
                baseline_y = global_y * (1.0 - surface_authority) + surface_y * surface_authority
                baseline_c = global_c * (1.0 - surface_authority) + surface_c * surface_authority
                baseline_amp_y = global_amp_y * (1.0 - surface_authority) + surface_amp_y * surface_authority
                baseline_amp_c = global_amp_c * (1.0 - surface_authority) + surface_amp_c * surface_authority
                baseline_ev_y = global_ev_y * (1.0 - surface_authority) + surface_ev_y * surface_authority
                baseline_ev_c = global_ev_c * (1.0 - surface_authority) + surface_ev_c * surface_authority

                # Secondary periods are intentionally absent from the old v4
                # surface fitter.  Add only their cycle-consensus fields here;
                # each already contains its own recurrence/reference evidence.
                for secondary_period in active_periods[1:]:
                    sec_y, sec_ay, sec_ey = _pwm_cycle_consensus_field(
                        logy_res, analysis_support, period=float(secondary_period),
                        sigma_x_ratio=adaptive_x_ratio, max_harmonics=4,
                        reference_correction=pwm_reference_luma,
                    )
                    sec_c, sec_ac, sec_ec = _pwm_cycle_consensus_field(
                        c_res, analysis_support, period=float(secondary_period),
                        sigma_x_ratio=adaptive_x_ratio, max_harmonics=4,
                        reference_correction=pwm_reference_chroma,
                    )
                    baseline_y = baseline_y + sec_y
                    baseline_c = baseline_c + sec_c
                    fallback_y = fallback_y + sec_y
                    fallback_c = fallback_c + sec_c
                    baseline_amp_y = torch.maximum(baseline_amp_y, sec_ay)
                    baseline_amp_c = torch.maximum(baseline_amp_c, sec_ac)
                    baseline_ev_y = torch.maximum(baseline_ev_y, sec_ey)
                    baseline_ev_c = torch.maximum(baseline_ev_c, sec_ec)

                field_y, seg_amp_y, seg_ev_y = _pwm_segmented_multisource_field(
                    logy_res, analysis_support, periods=active_periods,
                    baseline_field=baseline_y, fallback_field=fallback_y, surface_guide=surface_guide,
                    reference_correction=pwm_reference_luma, max_regions=12,
                )
                field_c, seg_amp_c, seg_ev_c = _pwm_segmented_multisource_field(
                    c_res, analysis_support, periods=active_periods,
                    baseline_field=baseline_c, fallback_field=fallback_c, surface_guide=surface_guide,
                    reference_correction=pwm_reference_chroma, max_regions=12,
                )
                amp_y = torch.maximum(baseline_amp_y, seg_amp_y)
                amp_c = torch.maximum(baseline_amp_c, seg_amp_c)
                evidence_y = torch.maximum(baseline_ev_y, seg_ev_y)
                evidence_c = torch.maximum(baseline_ev_c, seg_ev_c)
            else:
                field_y = baseline_y.mean(dim=-1, keepdim=True).expand_as(baseline_y)
                field_c = baseline_c.mean(dim=-1, keepdim=True).expand_as(baseline_c)
                amp_y = torch.ones_like(y)
                amp_c = torch.ones_like(y)
                evidence_y = torch.ones_like(y)
                evidence_c = torch.ones_like(y)

            if phase_lock_used:
                field_y = phase_lock_y
                field_c = phase_lock_c
                amp_y = phase_lock_amp_y
                amp_c = phase_lock_amp_c
                evidence_y = phase_lock_ev_y
                evidence_c = phase_lock_ev_c
        else:
            # Residual-only fallback retains the older absolute waveform levels
            # but still uses X-only amplitude rather than the smooth 2-D fitter.
            if adaptive:
                amp_y, evidence_y = _pwm_xonly_amplitude(
                    logy_res, corr_y_map, analysis_support, edge_support,
                    period=period,
                    sigma_x_ratio=adaptive_x_ratio,
                    corr_low=min(float(adaptive_corr_low), 0.005),
                    corr_high=min(float(adaptive_corr_high), 0.045),
                    max_gain=adaptive_max_gain,
                )
                amp_c, evidence_c = _pwm_xonly_amplitude(
                    c_res, corr_c_map, analysis_support, edge_support,
                    period=period,
                    sigma_x_ratio=adaptive_x_ratio,
                    corr_low=min(float(adaptive_corr_low), 0.005),
                    corr_high=min(float(adaptive_corr_high), 0.045),
                    max_gain=adaptive_max_gain,
                )
            else:
                amp_y = torch.ones_like(y)
                amp_c = torch.ones_like(y)
                evidence_y = torch.ones_like(y)
                evidence_c = torch.ones_like(y)
            field_y = corr_y_map * amp_y
            field_c = corr_c_map * amp_c.expand(-1, 2, -1, -1)

        # Remove DC from the *actual* X-adaptive fields.  PWM correction may
        # change the state contrast but must not become a global exposure or
        # white-balance control.
        field_y = field_y - (field_y * apply_y).sum(dim=(-2, -1), keepdim=True) / apply_y.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        field_c = field_c - (field_c * apply_c).sum(dim=(-2, -1), keepdim=True) / apply_c.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

        if no_harm and not pwm_advanced:
            # The residual-only fallback has no independent neural timing cue, so
            # retain a separate template-projection veto.  In the preferred
            # neural-edge-locked path, _pwm_xonly_direct_field already performs a
            # signed least-squares fit with local correlation/coverage evidence;
            # applying a second no-harm gain here double-attenuates valid PWM
            # correction on textured subjects.  Its no-harm policy is therefore
            # integrated into that direct fit rather than multiplied in twice.
            proposed_y = apply_y * field_y * float(luma_strength)
            proposed_c = apply_c * field_c * float(chroma_strength)
            safe_y = _pwm_profile_no_harm_gate(
                logy_res, proposed_y, analysis_support, edge_support,
                period=period, base_ratio=working_base_ratio,
            )
            safe_c = _pwm_profile_no_harm_gate(
                c_res, proposed_c, analysis_support, edge_support,
                period=period, base_ratio=working_base_ratio,
            )
            apply_y = (apply_y * safe_y).clamp(0.0, 1.0)
            apply_c_scalar = (apply_c_scalar * safe_c).clamp(0.0, 1.0)
            apply_c = apply_c_scalar.expand(-1, 2, -1, -1)
            evidence_y = (evidence_y * safe_y).clamp(0.0, 1.0)
            evidence_c = (evidence_c * safe_c).clamp(0.0, 1.0)

    else:
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
                narrow_ratio=working_narrow_ratio,
                base_ratio=working_base_ratio,
            )
            amp_c, evidence_c = _adaptive_profile_amplitude(
                c_res, corr_c_map, support, edge_support,
                period=period,
                sigma_x_ratio=adaptive_x_ratio,
                sigma_y_ratio=adaptive_y_ratio,
                corr_low=adaptive_corr_low,
                corr_high=adaptive_corr_high,
                max_gain=adaptive_max_gain,
                narrow_ratio=working_narrow_ratio,
                base_ratio=working_base_ratio,
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
                period=period, narrow_ratio=working_narrow_ratio, base_ratio=working_base_ratio,
            )
            safe_c = _profile_no_harm_gate(
                c_res, proposed_c, support, edge_support,
                period=period, narrow_ratio=working_narrow_ratio, base_ratio=working_base_ratio,
            )
            apply_y = (apply_y * safe_y).clamp(0.0, 1.0)
            apply_c_scalar = (apply_c_scalar * safe_c).clamp(0.0, 1.0)
            apply_c = apply_c_scalar.expand(-1, 2, -1, -1)
            evidence_y = (evidence_y * safe_y).clamp(0.0, 1.0)
            evidence_c = (evidence_c * safe_c).clamp(0.0, 1.0)

    # Bright soft-edge safeguard for intentionally aggressive profile strengths.
    # White flowers, masks and pale defocused boundaries can sit outside the
    # dominant-wall owner but still have almost no reliable local evidence for
    # a 1.8-2.0x row correction.  Limit only high-strength use near such bright
    # scene boundaries; darker skin/face interiors remain untouched.
    if (not pwm_active) and high_strength_guard is not None:
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
    if (not pwm_active) and surface_handoff is not None and float(surface_strength_cap) >= 0.0:
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

    effective_luma_strength = float(luma_strength)
    effective_chroma_strength = float(chroma_strength)
    if profile_mode == "pwm" and 'phase_lock_used' in locals() and phase_lock_used:
        # A phase-locked direct fit already estimates the full residual amplitude.
        # Treat >1 as a modest creative overdrive rather than multiplying a full
        # physical estimate by 2x+ and flipping the bands.  Other PWM modes retain
        # the historical strength semantics.
        effective_luma_strength = min(effective_luma_strength, 1.15)
        effective_chroma_strength = min(effective_chroma_strength, 1.15)

    applied_log_y = apply_y * field_y * effective_luma_strength
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
    out_c = c.float() + apply_c * field_c * effective_chroma_strength
    y_rms = float(applied_log_y.square().mean().sqrt())
    c_rms = float((apply_c * field_c * effective_chroma_strength).square().mean().sqrt())
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
    residual_profile_mode: str = "smooth",
    residual_profile_narrow_ratio: float = 0.035,
    residual_profile_base_ratio: float = 0.40,
    residual_profile_pwm_transition_ratio: float = 0.010,
    residual_profile_pwm_min_duty: float = 0.08,
    residual_profile_pwm_max_duty: float = 0.92,
    residual_profile_pwm_min_transition_score: float = 2.0,
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
    residual_profile_pwm_polish: bool = False,
    residual_profile_pwm_polish_strength: float = 1.0,
    residual_profile_pwm_polish_passes: int = 2,
    surface_equalizer_enabled: bool = False,
    surface_equalizer_mode: str = "consensus",
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
    broad_consensus_regions: int = 6,
    broad_consensus_min_regions: int = 2,
    broad_consensus_corr_low: float = 0.55,
    broad_consensus_corr_high: float = 0.80,
    broad_consensus_smooth_fraction: float = 0.015,
    broad_consensus_baseline_fraction: float = 0.20,
    broad_neural_gain_hint: torch.Tensor | None = None,
    broad_neural_chroma_hint: torch.Tensor | None = None,
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
    residual_profile_mode = str(residual_profile_mode).lower()
    if residual_profile_mode not in {"smooth", "pwm"}:
        raise ValueError("residual_profile_mode must be 'smooth' or 'pwm'")
    if residual_profile_pwm_transition_ratio <= 0:
        raise ValueError("PWM transition ratio must be > 0")
    if not (0.0 < residual_profile_pwm_min_duty < residual_profile_pwm_max_duty < 1.0):
        raise ValueError("PWM duty limits must satisfy 0 < min < max < 1")
    if residual_profile_pwm_min_transition_score <= 0:
        raise ValueError("PWM minimum transition score must be > 0")
    if residual_profile_adaptive_x_ratio <= 0 or residual_profile_adaptive_y_ratio <= 0:
        raise ValueError("adaptive profile x/y ratios must be > 0")
    if not (0.0 <= residual_profile_adaptive_corr_low < residual_profile_adaptive_corr_high <= 1.0):
        raise ValueError("adaptive profile correlation thresholds must satisfy 0 <= low < high <= 1")
    if residual_profile_adaptive_max_gain <= 0:
        raise ValueError("residual_profile_adaptive_max_gain must be > 0")
    if residual_profile_pwm_polish_strength < 0 or residual_profile_pwm_polish_strength > 1.25:
        raise ValueError("residual_profile_pwm_polish_strength must be in [0, 1.25]")
    if not (1 <= int(residual_profile_pwm_polish_passes) <= 6):
        raise ValueError("residual_profile_pwm_polish_passes must be in [1, 6]")
    surface_equalizer_mode = str(surface_equalizer_mode).lower()
    if surface_equalizer_mode not in {"dominant", "consensus"}:
        raise ValueError("surface_equalizer_mode must be 'dominant' or 'consensus'")
    if broad_consensus_regions < 2:
        raise ValueError("broad_consensus_regions must be >= 2")
    if not (2 <= broad_consensus_min_regions <= broad_consensus_regions):
        raise ValueError("broad_consensus_min_regions must be between 2 and broad_consensus_regions")
    if not (0.0 <= broad_consensus_corr_low < broad_consensus_corr_high <= 1.0):
        raise ValueError("broad consensus correlations must satisfy 0 <= low < high <= 1")
    if broad_consensus_smooth_fraction <= 0 or broad_consensus_baseline_fraction <= broad_consensus_smooth_fraction:
        raise ValueError("broad consensus fractions must satisfy 0 < smooth < baseline")

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
    # PWM / Step has two independent timing cues.  The cumulative neural
    # correction is usually the cleanest source because scene structure cancels
    # in source->restored deltas.  The post-neural image supplies a second,
    # derivative/coherence cue that is especially useful when the last network
    # pass is weak or a harmonic wins the ordinary spectrum.  Manual periods
    # remain exact.
    pwm_period_source = "none"
    residual_pwm_period_est = None
    if (
        profile_enabled
        and residual_profile_mode == "pwm"
        and residual_profile_band_period_px <= 0
        and band_period_px <= 0
    ):
        neural_pwm_period_est = _pwm_neural_edge_period(
            luma_field, chroma_delta_hint,
            full_height=h,
            min_period_px=period_min_px,
            max_period_fraction=period_max_fraction,
        )
        residual_pwm_period_est = _pwm_residual_edge_period(
            y,
            full_height=h,
            min_period_px=period_min_px,
            max_period_fraction=period_max_fraction,
        )
        if neural_pwm_period_est is None and residual_pwm_period_est is not None:
            period_est = residual_pwm_period_est
            pwm_period_source = "residual"
        elif neural_pwm_period_est is not None and residual_pwm_period_est is None:
            period_est = neural_pwm_period_est
            pwm_period_source = "neural"
        elif neural_pwm_period_est is not None and residual_pwm_period_est is not None:
            pn = float(neural_pwm_period_est.period_px)
            pr = float(residual_pwm_period_est.period_px)
            cn = float(neural_pwm_period_est.confidence)
            cr = float(residual_pwm_period_est.confidence)
            ratio = max(pn, pr) / max(1e-6, min(pn, pr))
            # Near-equal estimates: use the visible residual timing once it has
            # moderate confidence, otherwise keep the cleaner neural cue.
            if abs(math.log(max(pn, 1e-6) / max(pr, 1e-6))) < math.log(1.06) and cr >= 0.22:
                period_est = residual_pwm_period_est
                pwm_period_source = "residual"
            else:
                # Harmonic ambiguity: prefer the smaller coherent period when the
                # larger estimate is near 2x/3x/4x and the shorter cue is not weak.
                harmonic = any(abs(ratio - k) < 0.08 * k for k in (2.0, 3.0, 4.0))
                shorter = residual_pwm_period_est if pr < pn else neural_pwm_period_est
                shorter_conf = cr if pr < pn else cn
                if harmonic and shorter_conf >= 0.20:
                    period_est = shorter
                    pwm_period_source = "residual" if shorter is residual_pwm_period_est else "neural"
                elif cr > cn + 0.15:
                    period_est = residual_pwm_period_est
                    pwm_period_source = "residual"
                else:
                    period_est = neural_pwm_period_est
                    pwm_period_source = "neural"

    profile_period = float(period_est.period_px)
    primary_phase_diag = (0.0, 0.0, 0, 0.0)
    if (
        profile_enabled and residual_profile_mode == "pwm"
        and residual_profile_band_period_px <= 0 and band_period_px <= 0
    ):
        # P3b: ungated closed-form period refinement.
        #
        # _pwm_phase_lock_diagnostics projects over the WHOLE band axis, so a
        # period error of dP/P drifts the phase by N*dP/P cycles and the
        # projection lands in a sinc null.  Measured: singerpwm (55 cycles)
        # amplitude 0.110 -> 0.020 at a 2% period error, while test1_input
        # (7 cycles) barely moves.  _pwm_refine_phase_locked_period gates on
        # that amplitude, so on high-cycle images the refinement that would fix
        # the period can only run once the period is already correct.
        #
        # Block the band axis into ~6-cycle chunks, where the projection
        # survives a much larger error, and fit the phase slope.  Closed form,
        # no gate, no search span.
        _p3b_residual = torch.log(y.float().clamp_min(0.0) + 0.02)
        _p3b_period, _p3b_blocks, _p3b_rms = refine_period_phase_slope(
            _p3b_residual, period_support, period=profile_period,
        )
        if (
            _p3b_blocks >= 3
            and _p3b_rms < 0.06
            and abs(math.log(max(_p3b_period, 1e-6) / max(profile_period, 1e-6))) < math.log(1.12)
        ):
            # Few blocks => weak slope leverage. Keep the refinement only if it
            # actually explains more band energy than the coarse estimate.
            _p3b_prof = _p3b_residual[0, 0].median(dim=-1).values
            _p3b_prof = _p3b_prof - _p3b_prof.mean()
            _p3b_new = coherent_mode_power(_p3b_prof, _p3b_period)
            _p3b_old = coherent_mode_power(_p3b_prof, profile_period)
            print(f"[P3b] coarse={profile_period:.4f} -> slope={_p3b_period:.4f} "
                  f"blocks={_p3b_blocks} rms={_p3b_rms:.4f} "
                  f"power {_p3b_old:.4g} -> {_p3b_new:.4g} "
                  f"{'ACCEPT' if _p3b_new > _p3b_old else 'REJECT'}")
            if _p3b_new > _p3b_old:
                profile_period = float(_p3b_period)

        # Now runs at the CORRECTED period, so primary_phase_diag reports honest
        # coherence.  It feeds strong_primary and the PWM polish gate; measured
        # at the old period those gates stay shut regardless of the above.
        refined_period, primary_phase_diag = _pwm_refine_phase_locked_period(
            y, period_support, period=profile_period, reference_correction=luma_field,
        )
        if abs(math.log(max(refined_period, 1e-6) / max(profile_period, 1e-6))) < math.log(1.06):
            profile_period = float(refined_period)
            if primary_phase_diag[0] >= 0.88 and primary_phase_diag[2] >= 3:
                period_est = BandPeriodEstimate(
                    profile_period, max(float(period_est.confidence), float(primary_phase_diag[0])),
                    ((profile_period, max(float(period_est.confidence), float(primary_phase_diag[0]))),)
                    + tuple(period_est.candidates),
                )

    pwm_transition_phases = None
    if profile_enabled and residual_profile_mode == "pwm":
        neural_phases = _pwm_neural_transition_phases(
            luma_field, chroma_delta_hint,
            full_height=h,
            period=profile_period,
            min_duty=residual_profile_pwm_min_duty,
            max_duty=residual_profile_pwm_max_duty,
        )
        residual_phases = _pwm_residual_transition_phases(
            y,
            full_height=h,
            period=profile_period,
            min_duty=residual_profile_pwm_min_duty,
            max_duty=residual_profile_pwm_max_duty,
        )
        # When the residual edge train is trustworthy, align the template to the
        # artifact that is actually still visible.  Otherwise retain the neural
        # phase cue.
        if residual_phases is not None and (
            neural_phases is None
            or pwm_period_source == "residual"
            or float(residual_phases[2]) >= max(0.25, 0.85 * float(neural_phases[2]))
        ):
            pwm_transition_phases = residual_phases
        else:
            pwm_transition_phases = neural_phases

    # v5 multi-source discovery.  Manual profile periods remain single-source and
    # exact; automatic PWM mode may add up to two non-harmonic regional families.
    pwm_periods: tuple[float, ...] = (float(profile_period),)
    if (
        profile_enabled and residual_profile_mode == "pwm"
        and residual_profile_band_period_px <= 0 and band_period_px <= 0
    ):
        # Primary-first v8 logic: if one period is independently phase-locked
        # across most of the scene, do not let broad autocorrelation/spectral
        # side peaks prevent the low-degree-of-freedom v6 path.  Only ambiguous
        # primaries proceed to multi-source discovery.  A weak neural reference
        # is explicitly allowed; when Restormer is disabled ref_ratio is 0.
        coh, amp, votes, ref_ratio = primary_phase_diag
        strong_primary = bool(
            coh >= 0.88 and votes >= 3 and amp >= 0.010
            and (luma_field is None or ref_ratio <= 0.35)
        )
        if not strong_primary:
            pwm_periods = _pwm_discover_periods_multiregion(
                luma_field, chroma_delta_hint, y, full_height=h,
                primary_period=float(profile_period),
                min_period_px=period_min_px, max_period_fraction=period_max_fraction,
                max_sources=3,
            )

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
        user_mask=profile_user_mask,
        high_strength_guard=torch.minimum(edge_support, broad_structure_support),
        period=profile_period,
        period_confidence=period_est.confidence,
        luma_strength=residual_profile_luma_strength,
        chroma_strength=residual_profile_chroma_strength,
        profile_mode=residual_profile_mode,
        narrow_ratio=residual_profile_narrow_ratio,
        base_ratio=residual_profile_base_ratio,
        pwm_transition_ratio=residual_profile_pwm_transition_ratio,
        pwm_min_duty=residual_profile_pwm_min_duty,
        pwm_max_duty=residual_profile_pwm_max_duty,
        pwm_min_transition_score=residual_profile_pwm_min_transition_score,
        pwm_transition_phases=pwm_transition_phases,
        pwm_refine_auto_timing=(float(residual_profile_band_period_px) <= 0.0),
        pwm_reference_luma=luma_field,
        pwm_reference_chroma=chroma_delta_hint,
        pwm_periods=pwm_periods,
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

    # v7 optional final PWM polish.  It reuses only period families already
    # validated by the main residual-profile stage and therefore cannot invent
    # a new band frequency.  The pass runs after both main profile and local
    # cleanup, so it measures the *actual* final residual those stages left.
    polish_y_rms = 0.0
    polish_c_rms = 0.0
    polish_passes_done = 0
    polish_improvement = 0.0
    polish_applied_y = torch.zeros_like(y)
    polish_applied_c = torch.zeros_like(c)
    if (
        bool(residual_profile_pwm_polish)
        and profile_enabled
        and residual_profile_mode == "pwm"
        and tuple(float(p) for p in pwm_periods if float(p) > 4.0)
    ):
        out_y, out_c, polish_y_rms, polish_c_rms, polish_passes_done, polish_improvement, polish_applied_y, polish_applied_c = _pwm_residual_polish(
            out_y, out_c,
            periods=tuple(float(p) for p in pwm_periods if float(p) > 4.0),
            support=profile_support, edge_support=edge_support, user_mask=profile_user_mask,
            luma_strength=float(residual_profile_pwm_polish_strength),
            chroma_strength=float(residual_profile_pwm_polish_strength),
            max_passes=int(residual_profile_pwm_polish_passes), eps=0.02,
        )

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
    broad_consensus_confidence = 0.0
    broad_consensus_regions_used = 0
    broad_consensus_evidence = torch.zeros_like(y)
    broad_consensus_chroma_evidence = torch.zeros_like(y)
    if surface_equalizer_enabled:
        before_eq_y, before_eq_c = out_y, out_c
        if surface_equalizer_mode == "consensus":
            eq_out_y, eq_y_rms, equalizer_y, equalizer_apply_y, broad_consensus_evidence, broad_y_confidence, broad_y_regions = _broad_consensus_row_equalizer(
                out_y,
                neural_gain_hint=broad_neural_gain_hint,
                edge_support=edge_support, raw_tone_support=raw_tone_support,
                luma_strength=surface_equalizer_luma_strength,
                region_count=broad_consensus_regions, min_regions=broad_consensus_min_regions,
                corr_low=broad_consensus_corr_low, corr_high=broad_consensus_corr_high,
                smooth_fraction=broad_consensus_smooth_fraction, baseline_fraction=broad_consensus_baseline_fraction,
                poly_degree=surface_equalizer_poly_degree, row_sigma=surface_equalizer_row_sigma,
                huber_k=surface_equalizer_huber_k, min_coverage=surface_equalizer_min_coverage,
            )
            if broad_neural_chroma_hint is not None and surface_equalizer_chroma_strength > 0:
                (
                    eq_out_c, eq_c_rms, equalizer_c, equalizer_apply_c,
                    broad_consensus_chroma_evidence, broad_c_confidence, broad_c_regions,
                ) = _broad_neural_guided_chroma_consensus_equalizer(
                    out_c,
                    broad_neural_chroma_hint,
                    edge_support=edge_support, raw_tone_support=raw_tone_support,
                    chroma_strength=surface_equalizer_chroma_strength,
                    region_count=broad_consensus_regions, min_regions=broad_consensus_min_regions,
                    corr_low=broad_consensus_corr_low, corr_high=broad_consensus_corr_high,
                    smooth_fraction=broad_consensus_smooth_fraction,
                    row_sigma=surface_equalizer_row_sigma,
                    huber_k=surface_equalizer_huber_k, min_coverage=surface_equalizer_min_coverage,
                )
            else:
                eq_out_c = out_c
                eq_c_rms = 0.0
                equalizer_c = torch.zeros_like(c)
                equalizer_apply_c = torch.zeros_like(y)
                broad_c_confidence = 0.0
                broad_c_regions = 0
            equalizer_apply = torch.maximum(equalizer_apply_y, equalizer_apply_c)
            broad_consensus_confidence = max(float(broad_y_confidence), float(broad_c_confidence))
            broad_consensus_regions_used = max(int(broad_y_regions), int(broad_c_regions))
            surface_region = equalizer_apply
            surface_candidate = torch.maximum(broad_consensus_evidence, broad_consensus_chroma_evidence)
        else:
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
            "pwm_polish_y": polish_applied_y.detach(),
            "pwm_polish_c": polish_applied_c.detach(),
            "pwm_polish_improvement": torch.full_like(y, float(polish_improvement)).detach(),
            "surface_equalizer_candidate": surface_candidate.detach(),
            "surface_equalizer_region": surface_region.detach(),
            "surface_equalizer_apply": equalizer_apply.detach(),
            "surface_equalizer_y": equalizer_y.detach(),
            "surface_equalizer_c": equalizer_c.detach(),
            "broad_consensus_evidence": broad_consensus_evidence.detach(),
            "broad_consensus_chroma_evidence": broad_consensus_chroma_evidence.detach(),
            "broad_consensus_confidence": torch.full_like(y, float(broad_consensus_confidence)).detach(),
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
            broad_consensus_confidence=float(broad_consensus_confidence),
            broad_consensus_regions=int(broad_consensus_regions_used),
            band_candidates=period_est.candidates,
            pwm_periods=tuple(float(p) for p in pwm_periods),
            pwm_polish_y_rms=float(polish_y_rms),
            pwm_polish_c_rms=float(polish_c_rms),
            pwm_polish_passes=int(polish_passes_done),
            pwm_polish_improvement=float(polish_improvement),
        )
    return out_y, out_c, effective_blend, stats
