#!/usr/bin/env python3
"""Ungated, closed-form PWM period refinement for Flicker Suppressor.

Drop this beside ``flat_region_filter.py``. It replaces the gated
``_pwm_refine_phase_locked_period`` path with a refinement that cannot deadlock
on high-cycle images.

WHY THIS EXISTS
---------------
``_pwm_phase_lock_diagnostics`` projects the residual onto sin/cos over the
*entire* band axis. When the assumed period is wrong by dP/P, the phase drifts
by N*dP/P cycles across a frame containing N cycles, and the projection lands in
a sinc null. Measured on the test set:

    test1_input   (7 cycles)  : amplitude 0.234 -> 0.213 at 4% period error
    singerpwm    (55 cycles)  : amplitude 0.110 -> 0.020 at 2% period error
    jDOn4whY...  (46 cycles)  : amplitude 0.355 -> 0.043 at 2% period error

``_pwm_refine_phase_locked_period`` gates on that amplitude (``base[1] < 0.008``)
and on coherence (``base[0] < 0.84``) *before* refining, so on high-cycle images
the refinement that would fix the period can only run once the period is already
correct. Low-cycle images sail through, which is why they work.

METHOD
------
Split the band axis into blocks of a few cycles. Inside a short block the
projection survives a much larger period error. The measured phase advances
linearly across blocks when the period is wrong:

    phi(y) = phi_0 + 2*pi*y*(1/P_true - 1/P_assumed)

so a weighted linear fit of unwrapped phase against block position gives the
correction in closed form. Two or three iterations converge.

VALIDATION (numpy reference, on the tests.zip inputs)
-----------------------------------------------------
Starting from deliberately wrong periods of -6%, -3%, -1.5%, +1.5%, +3%, +6%,
the refined period was identical in every case:

    singerpwm        true 21.1536 -> 21.1525  (-0.005%)   9 blocks
    jDOn4whY...      true 12.1004 -> 12.1000  (-0.004%)   7 blocks
    test3_input      true 38.1465 -> 38.0666  (-0.209%)   4 blocks
    0-q9gicytxxn4h1  true 148.177 -> 148.749  (+0.386%)   4 blocks
    test1_input      true 96.3130 -> 96.1773  (-0.141%)   2 blocks

Required precision (phase drift under P/8 across the frame) is 0.23%, 0.27%,
0.64%, 0.45% and 1.81% respectively, so all but 0-q9gicytxxn4h1 clear it
comfortably; that one is marginal and benefits from ``fine_scan_period`` below.

STATUS: the numpy reference was validated on the real images. This torch
translation has NOT been executed - there is no PyTorch in the environment it
was written in. Treat it as a reviewed patch, not tested code. The numpy
reference is in ``bandmetrics.py`` if you want to cross-check outputs.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["refine_period_phase_slope", "fine_scan_period", "coherent_mode_power"]


def _smooth_y(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian along the band axis. Mirrors _smooth_axis(..., 'y')."""
    sigma = float(sigma)
    if sigma <= 0.0:
        return x
    radius = max(1, int(round(3.0 * sigma)))
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1, 1)
    c = x.shape[1]
    xp = F.pad(x, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(xp, k.expand(c, 1, -1, -1), groups=c)


def _block_phasor(
    hp: torch.Tensor,
    ww: torch.Tensor,
    y0: int,
    y1: int,
    omega: float,
) -> tuple[complex, float] | None:
    """Support-weighted complex amplitude of one band-axis block.

    Returns ``(phasor, median_amplitude)`` or None when the block has no usable
    support. Each perpendicular column contributes its own phasor; they are
    combined with amplitude-squared weighting so a bright coherent surface
    dominates a dim noisy one.
    """
    seg = hp[..., y0:y1, :]
    sup = ww[..., y0:y1, :]
    h = y1 - y0
    yy = torch.arange(y0, y1, device=hp.device, dtype=torch.float32).view(1, 1, h, 1)
    sn = torch.sin(yy * omega)
    cs = torch.cos(yy * omega)
    den = sup.sum(dim=-2).clamp_min(1e-6)
    aa = 2.0 * (seg * sup * sn).sum(dim=-2) / den
    bb = 2.0 * (seg * sup * cs).sum(dim=-2) / den
    amp = torch.sqrt(aa.square() + bb.square()).clamp_min(1e-12)
    cov = (den / float(h)).clamp(0.0, 1.0)
    valid = cov > 0.18
    if int(valid.sum()) < 3:
        return None
    w = (amp.square() * cov * valid.to(amp.dtype))
    tot = w.sum()
    if float(tot) <= 1e-12:
        return None
    sx = float((w * (aa / amp)).sum())
    sy = float((w * (bb / amp)).sum())
    med = float(amp[valid].median())
    return complex(sx, sy), med


def refine_period_phase_slope(
    residual: torch.Tensor,
    support: torch.Tensor,
    *,
    period: float,
    cycles_per_block: float = 6.0,
    min_blocks: int = 3,
    max_iter: int = 3,
    max_relative_move: float = 0.25,
) -> tuple[float, int, float]:
    """Refine ``period`` by fitting the phase slope across band-axis blocks.

    Deliberately ungated: when evidence is absent it returns the input period
    unchanged rather than refusing to run. Safe to call unconditionally.

    Returns ``(refined_period, n_blocks, phase_residual_cycles)``.
    ``phase_residual_cycles`` is the weighted RMS departure from a straight
    phase line, in cycles. Values under ~0.02 indicate a genuinely stationary
    single source; large values mean the period is drifting or more than one
    source is present, and the caller should be cautious.
    """
    p = float(period)
    if residual.ndim != 4 or residual.shape[1] != 1 or p <= 4.0:
        return p, 0, 0.0
    if support.ndim != 4 or support.shape[1] != 1:
        return p, 0, 0.0
    b, _, h, w = residual.shape
    if b != 1 or w < 16:
        return p, 0, 0.0

    x = residual.float()
    ww = support.float().clamp(0.0, 1.0)
    n_blocks = 0
    phase_rms = 0.0

    for _ in range(max(1, int(max_iter))):
        blk = max(int(round(cycles_per_block * p)), int(round(3.0 * p)))
        k = h // max(blk, 1)
        if k < min_blocks:
            blk = max(int(round(3.0 * p)), h // max(min_blocks, 1))
            k = h // max(blk, 1)
        if k < 2:
            return p, k, phase_rms

        hp = x - _smooth_y(x, max(8.0, 1.8 * p))
        omega = 2.0 * math.pi / p

        centers: list[float] = []
        phases: list[float] = []
        weights: list[float] = []
        for i in range(k):
            y0, y1 = i * blk, (i + 1) * blk
            got = _block_phasor(hp, ww, y0, y1, omega)
            if got is None:
                continue
            z, med = got
            centers.append(0.5 * (y0 + y1))
            phases.append(math.atan2(z.imag, z.real))
            weights.append(max(med, 1e-9))

        n_blocks = len(centers)
        if n_blocks < 2:
            return p, n_blocks, phase_rms

        # Unwrap. Blocks are ordered, so a simple sequential unwrap is enough.
        unwrapped = [phases[0]]
        for i in range(1, n_blocks):
            d = phases[i] - phases[i - 1]
            d = math.atan2(math.sin(d), math.cos(d))
            unwrapped.append(unwrapped[-1] + d)

        tw = sum(weights)
        if tw <= 1e-12:
            return p, n_blocks, phase_rms
        wn = [x_ / tw for x_ in weights]
        cbar = sum(wi * ci for wi, ci in zip(wn, centers))
        pbar = sum(wi * pi for wi, pi in zip(wn, unwrapped))
        num = sum(wi * (ci - cbar) * (pi - pbar) for wi, ci, pi in zip(wn, centers, unwrapped))
        den = sum(wi * (ci - cbar) ** 2 for wi, ci in zip(wn, centers))
        if den <= 1e-12:
            return p, n_blocks, phase_rms
        slope = num / den

        inv_new = 1.0 / p + slope / (2.0 * math.pi)
        if inv_new <= 0.0:
            return p, n_blocks, phase_rms
        p_new = 1.0 / inv_new
        if abs(p_new / p - 1.0) > float(max_relative_move):
            return p, n_blocks, phase_rms

        resid = [pi - (pbar + slope * (ci - cbar)) for ci, pi in zip(centers, unwrapped)]
        phase_rms = math.sqrt(sum(wi * r * r for wi, r in zip(wn, resid))) / (2.0 * math.pi)

        converged = abs(p_new / p - 1.0) < 1e-5
        p = p_new
        if converged:
            break

    return p, n_blocks, phase_rms


def coherent_mode_power(
    profile: torch.Tensor,
    period: float,
    *,
    nharm: int = 4,
) -> float:
    """Energy explained by a harmonic series at ``period`` on a 1-D profile.

    ``profile`` is a 1-D tensor sampled at FULL band-axis resolution. Used as
    the objective for ``fine_scan_period``.
    """
    v = profile.detach().float().flatten()
    n = int(v.numel())
    if n < 8 or period < 4.0:
        return 0.0
    t = torch.arange(n, device=v.device, dtype=torch.float32)
    total = 0.0
    for k in range(1, int(nharm) + 1):
        if period / k < 2.2:
            break
        omega = 2.0 * math.pi * k / float(period)
        re = float((v * torch.cos(t * omega)).sum())
        im = float((v * torch.sin(t * omega)).sum())
        total += re * re + im * im
    return total


def fine_scan_period(
    profile: torch.Tensor,
    period: float,
    *,
    span: float = 0.02,
    points: int = 401,
    nharm: int = 4,
    passes: int = 2,
) -> float:
    """Polish a period by maximising coherent harmonic power on a 1-D profile.

    Cheap: operates on the robust band profile, not the image. Use after
    ``refine_period_phase_slope`` when the phase residual is small and the image
    has many cycles (the case where required precision is tightest).
    """
    p = float(period)
    lo, hi = p * (1.0 - span), p * (1.0 + span)
    for _ in range(max(1, int(passes))):
        best_p, best_v = p, -1.0
        for i in range(int(points)):
            pp = lo + (hi - lo) * i / max(1, int(points) - 1)
            v = coherent_mode_power(profile, pp, nharm=nharm)
            if v > best_v:
                best_p, best_v = pp, v
        p = best_p
        half = (hi - lo) / 8.0
        lo, hi = p - half, p + half
    return p
