#!/usr/bin/env python3
"""tone_restore.py — restore gamma/contrast lost during processing.

WHY THIS EXISTS
---------------
Measured on band-axis-smoothed luminance, so the flicker itself does not
contaminate the statistic:

    image                        contrast    p99 shift   fitted gamma
    singerpwm  (--no-restormer)     -0.1%      -0.0004          1.001
    jDOn4whYSCFqB6NK023zIouFqVB     -8.6%      -0.0509          0.953
    pwm1                           -11.2%      -0.0314          0.882
    0-q9gicytxxn4h1                -13.3%      -0.0834          0.945
    test1_input                    -14.1%      -0.0418          0.821
    test3_input                    -22.4%      -0.1587          0.782

Every image that loses contrast runs the neural passes. singerpwm, the only one
with Restormer disabled, is untouched. test3_input runs two passes and loses
the most.

THE METHOD, AND WHY IT IS IN LOG SPACE
--------------------------------------
The obvious approach -- fit a tone curve on the deflickered envelopes and apply
it to full-resolution luminance -- does NOT work. A non-linear curve applied to
a still-banded image re-expands the residual band. Measured cost of that naive
version:

    test3_input   suppression 70.3% -> 66.6%
    pwm1          suppression  7.8% ->  0.6%     (nearly all correction undone)

Instead the correction is applied as a SMOOTH ADDITIVE OFFSET IN LOG SPACE:

    delta = log(mapped_envelope + eps) - log(envelope + eps)
    delta = smooth_along_band(delta, sigma = 0.6 * period)
    out   = exp(log(proc + eps) + delta) - eps

Because log(out + eps) = log(proc + eps) + delta exactly, and delta is smoothed
at sigma = 0.6*period (attenuating the fundamental by ~8e-4), the band residual
passes through untouched in log terms. Only the smooth envelope moves.

VALIDATED on the real test pairs, with periods correctly scaled to the analysis
resolution (an earlier run that skipped that scaling produced misleading
numbers):

    image                        supp before -> after    contrast before -> after
    singerpwm                        67.5      67.5          0.999      0.997
    jDOn4whYSCFqB6NK023zIouFqVB      43.5      43.5          0.914      0.992
    0-q9gicytxxn4h1                  38.0      37.9          0.863      0.965
    test3_input                      71.0      70.4          0.776      0.989
    test1_input                      43.0      42.1          0.859      0.994
    pwm1                             23.0      20.3          0.888      0.986

pwm1's -2.7 is the only real interference: it has no stable period, so
"smoothing along the band axis" smooths at a frequency that does not match what
is actually there. Expected, and it is already the image where nothing works.

The delta smoothing multiplier was swept at 0.6 / 1.5 / 3.0 periods. Heavier
smoothing buys almost nothing on suppression and costs real contrast accuracy
(test3_input 0.989 -> 0.942 at 3.0). 0.6 is the right value.

SAFETY
------
  * monotone by construction (cumulative max) -- no tone inversion
  * gain clamped to [1/max_gain, max_gain], default 1.6
  * strength=0 is an exact no-op
  * chroma untouched: this returns luminance only

STATUS: the numpy path was validated on the real images (numbers above). The
torch path has now been run on test3_input (contrast 0.776 -> 0.932, margin
26.0 -> 25.6, signed +0.110 -> +0.111) and hardened against torch.quantile's
~16M element limit, which 26 MP inputs exceed. It has NOT been exhaustively
exercised -- no PyTorch was available where this was
written. Review it as a patch, not as tested code.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:                                    # numpy-only use
    torch = None
    F = None

__all__ = ["match_tone_log", "measure_tone", "match_tone_log_torch",
           "DELTA_SMOOTH_PERIODS", "EPS"]

EPS = 0.02
DELTA_SMOOTH_PERIODS = 0.6      # swept: 0.6 best; 1.5/3.0 cost contrast accuracy


# --------------------------------------------------------------- numpy core
def _smooth_band(y: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    """1-D Gaussian blur along one axis, numpy only.

    scipy is deliberately avoided: this module ships inside the Nuitka build,
    and although only the torch path runs there, a function-level scipy import
    is still something Nuitka tries to resolve. Bundling scipy for one blur is
    not worth the size or the extension-module fragility.
    """
    sigma = max(1.5, float(sigma))
    radius = max(1, int(round(3.0 * sigma)))
    k = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (k / sigma) ** 2)
    k /= k.sum()
    pad = [(0, 0)] * y.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(y.astype(np.float64), pad, mode="edge")
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), axis, padded)


def _envelope(y: np.ndarray, period: float, axis: int) -> np.ndarray:
    if period <= 3.0:
        return y
    return _smooth_band(y, DELTA_SMOOTH_PERIODS * period, axis)


def _curve(src_env: np.ndarray, dst_env: np.ndarray, *, knots: int,
           max_gain: float, lo_q: float = 0.5, hi_q: float = 99.5):
    """Monotone lookup mapping dst envelope values onto src envelope values."""
    qs = np.linspace(lo_q, hi_q, knots)
    a = np.maximum.accumulate(np.percentile(src_env, qs))
    b = np.maximum.accumulate(np.percentile(dst_env, qs))
    b = b + np.arange(knots) * 1e-6            # strictly increasing domain
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(b > 1e-6, a / b, 1.0)
    gain = np.clip(gain, 1.0 / max_gain, max_gain)
    return b, np.maximum.accumulate(b * gain)


def _headroom_log(y: np.ndarray, knee_lo: float = 0.70, knee_hi: float = 0.90,
                  keep: float = 0.45) -> np.ndarray:
    """Largest positive log gain that leaves highlight headroom intact.

    Same idiom the residual profile stage already uses (flat_region_filter.py,
    "Preserve highlight headroom when a strong positive residual profile lands
    on bright foreground detail"). Without it the tone curve pushes near-white
    pixels past 1.0 and .clamp(0, 1) turns a graded lamp falloff into a flat
    white disc.

    Measured on 0-q9gicytxxn4h1: tone restore reported delta-max 0.2885
    (a 1.334x gain) and drove 5.06% of the lamp-core crop into NEW clipping.
    With strength=0 the same crop clipped 0.000%. Negative corrections are
    never limited -- only pushing up is dangerous.
    """
    t = np.clip((y - knee_lo) / max(1e-8, knee_hi - knee_lo), 0.0, 1.0)
    bright = t * t * (3.0 - 2.0 * t)
    max_y = y + (1.0 - keep * bright) * (1.0 - y)
    return np.log((max_y + EPS) / np.maximum(y + EPS, 1e-6))


def match_tone_log(dst: np.ndarray, src: np.ndarray, *, period: float = 0.0,
                   axis: int = 0, strength: float = 1.0, knots: int = 33,
                   max_gain: float = 1.6) -> np.ndarray:
    """Pull `dst` luminance onto `src`'s tone curve without touching the band.

    dst, src : HxW luminance in [0, 1].
    period, axis : describe the band so both the fit and the correction field
        are flicker-free. axis=0 for horizontal bands (varying down rows),
        axis=1 for vertical bands. The period must be expressed in the SAME
        pixel units as the arrays -- scale it if the arrays were resized.
    """
    if strength <= 0.0:
        return dst
    se = _envelope(src, period, axis)
    de = _envelope(dst, period, axis)
    xs, ys = _curve(se, de, knots=knots, max_gain=max_gain)
    mapped = np.interp(de, xs, ys)
    delta = np.log(mapped + EPS) - np.log(de + EPS)
    if period > 3.0:
        # Guarantees the correction field carries no band-frequency content.
        delta = _smooth_band(delta, DELTA_SMOOTH_PERIODS * period, axis)
    lim = np.log(max_gain)
    delta = np.clip(delta, -lim, lim)
    delta = float(strength) * delta
    # Never push a pixel toward clipping (see _headroom_log).
    delta = np.minimum(delta, _headroom_log(dst))
    out = np.exp(np.log(dst + EPS) + delta) - EPS
    return np.clip(out, 0.0, 1.0)


def measure_tone(dst: np.ndarray, src: np.ndarray, *, period: float = 0.0,
                 axis: int = 0) -> dict:
    """Contrast / gamma drift of dst relative to src, measured on envelopes."""
    se = _envelope(src, period, axis)
    de = _envelope(dst, period, axis)
    m = (se > 0.05) & (se < 0.95)
    gamma = float("nan")
    if m.sum() > 100:
        gamma = float(np.polyfit(np.log(se[m] + 1e-6), np.log(de[m] + 1e-6), 1)[0])
    return {
        "contrast_ratio": float(de.std() / max(se.std(), 1e-9)),
        "gamma": gamma,
        "mean_shift": float(de.mean() - se.mean()),
        "p1_shift": float(np.percentile(de, 1) - np.percentile(se, 1)),
        "p99_shift": float(np.percentile(de, 99) - np.percentile(se, 99)),
    }


# ---------------------------------------------------------------- torch path
def match_tone_log_torch(proc_y, ref_y, *, period: float = 0.0,
                         strength: float = 1.0, knots: int = 33,
                         max_gain: float = 1.6):
    """Bx1xHxW luminance in, corrected luminance + stats out.

    Band axis is rows (dim -2), matching processing orientation. UNTESTED.
    """
    if torch is None:
        raise RuntimeError("torch unavailable")
    if strength <= 0.0:
        return proc_y, {"contrast_ratio": 1.0, "delta_mean": 0.0, "delta_max": 0.0}

    def smooth(t):
        if period <= 3.0:
            return t
        sigma = max(1.5, DELTA_SMOOTH_PERIODS * float(period))
        radius = max(1, int(round(3.0 * sigma)))
        k = torch.arange(-radius, radius + 1, device=t.device, dtype=torch.float32)
        k = torch.exp(-0.5 * (k / sigma) ** 2)
        k = (k / k.sum()).view(1, 1, -1, 1)
        return F.conv2d(F.pad(t.float(), (0, 0, radius, radius), mode="replicate"), k)

    se, de = smooth(ref_y), smooth(proc_y)
    qs = torch.linspace(0.005, 0.995, knots, device=proc_y.device)

    def _q(t):
        """Quantiles with a size guard.

        torch.quantile refuses inputs beyond ~16M elements (0-q9gicytxxn4h1 is
        26M). Strided subsampling to at most 4M is statistically identical for
        33 knots and avoids a full sort of the whole image.
        """
        v = t.flatten()
        n = int(v.numel())
        cap = 4_000_000
        if n > cap:
            v = v[:: (n + cap - 1) // cap]
        return torch.quantile(v.float(), qs)

    a = torch.cummax(_q(se), dim=0).values
    b = torch.cummax(_q(de), dim=0).values
    b = b + torch.arange(knots, device=b.device, dtype=b.dtype) * 1e-6
    gain = torch.where(b > 1e-6, a / b, torch.ones_like(b)).clamp(1.0 / max_gain, max_gain)
    a = torch.cummax(b * gain, dim=0).values

    flat = de.flatten().contiguous()
    idx = torch.searchsorted(b.contiguous(), flat.clamp(b[0], b[-1]), right=True).clamp(1, knots - 1)
    x0, x1, y0, y1 = b[idx - 1], b[idx], a[idx - 1], a[idx]
    w = (flat - x0) / (x1 - x0).clamp_min(1e-9)
    mapped = (y0 + w * (y1 - y0)).view_as(de)

    delta = smooth(torch.log(mapped + EPS) - torch.log(de + EPS))
    lim = float(np.log(max_gain))
    delta = delta.clamp(-lim, lim) * float(strength)

    # Never push a pixel toward clipping. Mirrors the residual profile stage's
    # highlight-headroom limiter; without it the lamp cores in
    # 0-q9gicytxxn4h1 were driven past 1.0 and flattened to white discs.
    t = ((proc_y - 0.70) / 0.20).clamp(0.0, 1.0)
    bright = t * t * (3.0 - 2.0 * t)
    max_y = proc_y + (1.0 - 0.45 * bright) * (1.0 - proc_y)
    max_delta = torch.log((max_y + EPS) / (proc_y + EPS).clamp_min(1e-6))
    delta = torch.minimum(delta, max_delta)

    out = (torch.exp(torch.log(proc_y + EPS) + delta) - EPS).clamp(0.0, 1.0)

    with torch.no_grad():
        stats = {
            "contrast_ratio": float(smooth(out).std() / smooth(ref_y).std().clamp_min(1e-9)),
            "delta_mean": float(delta.mean()),
            "delta_max": float(delta.abs().max()),
            "headroom_limited": float((delta >= max_delta - 1e-6).float().mean()),
        }
    return out, stats
