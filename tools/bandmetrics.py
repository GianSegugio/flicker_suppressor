#!/usr/bin/env python3
"""Objective band-suppression metrics for Flicker Suppressor regression testing.

Requires only numpy + scipy + Pillow. No PyTorch, so it runs anywhere and can be
wired into CI without a GPU.

WHY YOU NEED THIS
-----------------
Global RMS of the row profile is a misleading metric: it is dominated by scene
structure, not by the artifact. On singerpwm it reported 0.7% suppression while
demodulation at the true period showed 66%. Every tuning decision made against
global RMS is being made against noise.

This module measures the thing you actually care about - energy at the flicker
frequency and its harmonics - and reports it against a scene-texture floor
measured at nearby control frequencies, so you can tell "removed the artifact"
apart from "smoothed the image".

USAGE
-----
    # single pair
    python bandmetrics.py --input in.jpg --output out.png --axis vertical

    # whole regression suite, writes CSV
    python bandmetrics.py --suite tests/ --csv results.csv

    # compare two runs (e.g. before/after a patch)
    python bandmetrics.py --suite tests/ --csv new.csv --baseline old.csv

READING THE OUTPUT
------------------
    supp%   suppression of band-mode amplitude, input -> output
    floor%  suppression at which only scene texture would remain
    margin  supp% - floor%. THIS is the number to optimise.
            <= 0 means the artifact was essentially untouched.
    ctrl%   change in control-frequency energy. Large positive values mean the
            correction is injecting new structure - reject the change.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import re

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d

Image.MAX_IMAGE_PIXELS = None
EPS = 0.02
CONTROL_RATIOS = (0.80, 0.86, 1.17, 1.24)


# --------------------------------------------------------------------- loading
def log_luma(path: Path, max_side: int = 2400) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        f = max_side / max(w, h)
        im = im.resize((int(round(w * f)), int(round(h * f))), Image.LANCZOS)
    a = np.asarray(im, np.float64) / 255.0
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return np.log(y + EPS)


def oriented(l: np.ndarray, axis: str) -> np.ndarray:
    """Return the array with the band axis first."""
    return l if axis == "horizontal" else l.T


def band_profile(l: np.ndarray, axis: str, max_perp: int = 1600) -> np.ndarray:
    """Robust 1-D profile at full band-axis resolution.

    Only the perpendicular axis is reduced. Reducing the band axis destroys the
    harmonics that distinguish a square wave from a sinusoid.
    """
    x = oriented(l, axis)
    if x.shape[1] > max_perp:
        k = int(np.ceil(x.shape[1] / max_perp))
        x = x[:, : (x.shape[1] // k) * k].reshape(x.shape[0], -1, k).mean(axis=2)
    p = np.median(x, axis=1)
    return p - gaussian_filter1d(p, max(4.0, len(p) * 0.04), mode="nearest")


# ------------------------------------------------------------- period estimate
def coherent_power(sig: np.ndarray, period: float, nharm: int = 4) -> float:
    n = len(sig)
    t = np.arange(n)
    s = 0.0
    for k in range(1, nharm + 1):
        if period / k < 2.2:
            break
        s += abs(np.sum(sig * np.exp(-2j * np.pi * k * t / period))) ** 2
    return float(s)


def harmonic_family_period(sig: np.ndarray, pmin: float = 6.0,
                           pmaxfrac: float = 0.25) -> float | None:
    """Coarse period by harmonic-family energy.

    Scores the whole family (P, P/2, P/3 ...) rather than taking the shortest
    lag above a ratio threshold. That threshold form picks up noise ripples on
    broad autocorrelation plateaus - it produced spurious 4-6 px periods on
    my_photo_input and ge107vg31 during development.
    """
    n = len(sig)
    m = np.abs(np.fft.rfft((sig - sig.mean()) * np.hanning(n)))
    f = np.fft.rfftfreq(n)
    ms = gaussian_filter1d(m, 1.0)
    noise = np.median(ms[2:])
    best, best_score = None, -1.0
    for i in range(2, len(m) - 1):
        if f[i] <= 0:
            continue
        p = 1.0 / f[i]
        if not (pmin <= p <= n * pmaxfrac):
            continue
        if not (ms[i] > ms[i - 1] and ms[i] >= ms[i + 1]):
            continue
        if ms[i] < 2.5 * noise:
            continue
        # prominence: must stand above the local trough, not just its neighbours
        lo = ms[max(2, i - 12):i].min() if i > 2 else ms[i]
        hi = ms[i + 1:min(len(ms), i + 13)].min() if i + 1 < len(ms) else ms[i]
        if ms[i] < 1.5 * max(lo, hi):
            continue
        score = 0.0
        for k in range(1, 7):
            j = int(round(i * k))
            if j >= len(m) - 1:
                break
            e = float(ms[max(0, j - 1):j + 2].max())
            if e > 1.5 * noise:
                score += e / k ** 0.5
        if score > best_score:
            best, best_score = p, score
    return best


def refine_period(sig: np.ndarray, p0: float, nharm: int = 4,
                  span: float = 0.12, points: int = 2001) -> float:
    lo, hi = p0 * (1 - span), p0 * (1 + span)
    p = p0
    for _ in range(2):
        ps = np.linspace(lo, hi, points)
        v = np.array([coherent_power(sig, q, nharm) for q in ps])
        p = float(ps[int(np.argmax(v))])
        half = (hi - lo) / 7.0
        lo, hi = p - half, p + half
    return p


def estimate_period(l: np.ndarray, axis: str) -> float | None:
    prof = band_profile(l, axis)
    p0 = harmonic_family_period(prof)
    if p0 is None:
        return None
    return refine_period(prof, p0)


def detect_axis(l: np.ndarray) -> str:
    """Pick the axis carrying real periodic energy, not the aspect ratio."""
    best, best_score = "horizontal", -1.0
    for ax in ("horizontal", "vertical"):
        prof = band_profile(l, ax)
        p = harmonic_family_period(prof)
        if p is None:
            continue
        score = coherent_power(prof, refine_period(prof, p)) / max(np.sum(prof ** 2) ** 2, 1e-12)
        if score > best_score:
            best, best_score = ax, score
    return best


# ------------------------------------------------------------------- measuring
def _envelope(y: np.ndarray, axis: str, period: float) -> np.ndarray:
    """Band-axis-smoothed luminance: the flicker-free envelope."""
    if period <= 3.0:
        return y
    ax = 0 if axis == "horizontal" else 1
    return gaussian_filter1d(y, max(1.5, 0.6 * period), axis=ax, mode="nearest")


def tone_drift(li: np.ndarray, lo: np.ndarray, axis: str, period: float) -> tuple:
    """Contrast ratio and fitted gamma of output vs input, on the envelope.

    Measured on envelopes so the flicker does not contaminate the statistic.
    contrast_ratio 1.000 = output contrast matches input; below 1 = flattened.
    """
    si = np.exp(li) - EPS
    so = np.exp(lo) - EPS
    a = _envelope(si, axis, period)
    b = _envelope(so, axis, period)
    m = (a > 0.05) & (a < 0.95)
    gamma = float("nan")
    if m.sum() > 100:
        gamma = float(np.polyfit(np.log(a[m] + 1e-6), np.log(b[m] + 1e-6), 1)[0])
    return float(b.std() / max(a.std(), 1e-9)), gamma


def band_phasor(l: np.ndarray, axis: str, period: float, harm: int = 1):
    """Complex local amplitude of one harmonic. Carries phase, unlike the modulus."""
    x = oriented(l, axis)
    n = x.shape[0]
    t = np.arange(n)[:, None]
    z = x * np.exp(-2j * np.pi * harm * t / period)
    s = period * 1.2
    return gaussian_filter(z.real, (s, s)) + 1j * gaussian_filter(z.imag, (s, s))


def signed_residual(li: np.ndarray, lo: np.ndarray, axis: str, period: float,
                    crop: float = 0.06):
    """Project the output residual onto the INPUT band phase.

    |amplitude| cannot separate an under-corrected residual from one driven past
    zero and inverted -- both read the same. This can:

        +1.0  untouched
         0.0  corrected
        <0    INVERTED, i.e. over-corrected

    Returns (mean signed ratio, fraction of area inverted past -0.15).
    """
    zi = band_phasor(li, axis, period)
    zo = band_phasor(lo, axis, period)
    mag = np.abs(zi)
    ref = zi / np.maximum(mag, 1e-9)
    proj = np.real(zo * np.conj(ref))
    n, w = proj.shape
    i0, i1 = int(n * crop), int(n * (1 - crop))
    j0, j1 = int(w * crop), int(w * (1 - crop))
    p, m = proj[i0:i1, j0:j1], mag[i0:i1, j0:j1]
    ratio = float(p.mean() / max(m.mean(), 1e-12))
    over = float((p < -0.15 * m).mean())
    return ratio, over


def mode_amplitude(l: np.ndarray, axis: str, period: float,
                   harms=(1, 2, 3), crop: float = 0.06) -> float:
    """Mean local band amplitude via complex demodulation."""
    x = oriented(l, axis)
    n, w = x.shape
    t = np.arange(n)[:, None]
    total = 0.0
    for k in harms:
        if period / k < 2.5:
            break
        z = x * np.exp(-2j * np.pi * k * t / period)
        zr = gaussian_filter(z.real, (period * 1.2, period * 1.0))
        zi = gaussian_filter(z.imag, (period * 1.2, period * 1.0))
        total = total + (2 * np.abs(zr + 1j * zi)) ** 2
    a = np.sqrt(total)
    i0, i1 = int(n * crop), int(n * (1 - crop))
    j0, j1 = int(w * crop), int(w * (1 - crop))
    return float(a[i0:i1, j0:j1].mean())


def prepare_source(src: Path, axis: str | None, period_orig: float | None):
    """Input-side analysis, shared by every output derived from this input.

    The suite has ~24 distinct inputs but ~35 outputs, and several inputs have
    five variants each. Computing ei/floor once per input rather than once per
    pair removes roughly half the total work.
    """
    li = log_luma(src)
    ax = axis or detect_axis(li)
    if period_orig is not None:
        W, H = Image.open(src).size
        scale = min(1.0, 2400.0 / max(W, H))
        p = float(period_orig) * scale
    else:
        p = estimate_period(li, ax)
    if p is None or p < 5.0:
        return None
    ei = mode_amplitude(li, ax, p)
    floor = float(np.mean([mode_amplitude(li, ax, p * r) for r in CONTROL_RATIOS]))
    return {"li": li, "ax": ax, "p": p, "ei": ei, "floor": floor}


def evaluate(src: Path, dst: Path, axis: str | None = None,
             period_orig: float | None = None, tier: str = "auto",
             prepared: dict | None = None) -> dict:
    """period_orig is in ORIGINAL image pixels and is scaled internally."""
    pre = prepared or prepare_source(src, axis, period_orig)
    if pre is None:
        return {"image": src.name, "error": "no period found"}
    li, ax, p, ei, floor = pre["li"], pre["ax"], pre["p"], pre["ei"], pre["floor"]
    lo_ = log_luma(dst)
    if li.shape != lo_.shape:
        return {"image": src.name, "error": f"shape mismatch {li.shape} vs {lo_.shape}"}

    eo = mode_amplitude(lo_, ax, p)
    ci = floor
    co = float(np.mean([mode_amplitude(lo_, ax, p * r) for r in CONTROL_RATIOS]))

    supp = 100.0 * (1 - eo / max(ei, 1e-12))
    floor_pct = 100.0 * (1 - floor / max(ei, 1e-12))
    signed, over = signed_residual(li, lo_, ax, p)
    contrast, gamma = tone_drift(li, lo_, ax, p)
    W, H = Image.open(src).size
    scale = min(1.0, 2400.0 / max(W, H))
    return {
        "image": src.name,
        "axis": ax,
        "tier": tier,
        "period_px": round(p / scale, 4),
        "cycles": round(oriented(li, ax).shape[0] / p, 1),
        "amp_in": round(ei, 5),
        "amp_out": round(eo, 5),
        "supp_pct": round(supp, 1),
        "floor_pct": round(floor_pct, 1),
        "margin": round(supp - floor_pct, 1),
        "ctrl_pct": round(100.0 * (co / max(ci, 1e-12) - 1), 1),
        "signed": round(signed, 3),
        "over_pct": round(100.0 * over, 1),
        "contrast": round(contrast, 3),
        "gamma": round(gamma, 3),
    }


# ----------------------------------------------------------------------- suite
def _analyse_group(job):
    """One input plus all outputs derived from it. Runs in a worker process."""
    src, axis, period_orig, outputs = job
    src = Path(src)
    try:
        pre = prepare_source(src, axis, period_orig)
    except Exception as e:                      # noqa: BLE001 - report, do not abort the run
        return [{"image": src.name, "output": Path(o[0]).name, "error": repr(e)}
                for o in outputs]
    rows = []
    for dst, tier in outputs:
        dst = Path(dst)
        try:
            r = evaluate(src, dst, axis, period_orig, tier, prepared=pre)
        except Exception as e:                  # noqa: BLE001
            r = {"image": src.name, "error": repr(e)}
        r["output"] = dst.name
        rows.append(r)
    return rows


def find_pairs(root: Path):
    ins, outs = root / "inputs", root / "restored"
    if not ins.is_dir() or not outs.is_dir():
        raise SystemExit(f"expected {ins} and {outs}")
    for o in sorted(outs.glob("*.png")):
        m = re.match(r"^(?P<base>.+?)_restored(?:_v\d+)?$", o.stem)
        if not m:
            continue
        base = m.group("base")
        cand = list(ins.glob(base + ".*"))
        if not cand:
            continue
        axis = None
        j = o.with_suffix(".json")
        if j.exists():
            try:
                a = json.load(open(j)).get("band_axis")
                axis = a if a in ("horizontal", "vertical") else None
            except Exception:
                pass
        yield cand[0], o, axis, base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--axis", choices=["horizontal", "vertical"])
    ap.add_argument("--period", type=float)
    ap.add_argument("--suite", help="directory containing inputs/ and restored/")
    ap.add_argument("--csv")
    ap.add_argument("--baseline", help="CSV from a previous run, to diff against")
    ap.add_argument("--pins", help="pins.json: fixed axis/period per image (STRONGLY recommended)")
    ap.add_argument("--tier", action="append",
                    help="only measure these tiers (repeatable), e.g. --tier target")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel worker processes (0 = auto: cpu_count-2, capped at 12). "
                         "Each worker holds a few large float64 arrays, so back this off "
                         "if memory is tight.")
    a = ap.parse_args()

    pins = {}
    if a.pins:
        pins = json.load(open(a.pins)).get("pins", {})
    want_tiers = set(a.tier) if a.tier else None

    rows = []
    if a.suite:
        # Group outputs by input so the input-side analysis is done once.
        groups: dict[str, list] = {}
        meta: dict[str, tuple] = {}
        for src, dst, ax, base in find_pairs(Path(a.suite)):
            pin = pins.get(src.name)
            if pin:
                tier = pin.get("tier", "auto")
                axis, per = pin["axis"], pin.get("period_px")
            else:
                if pins:
                    print(f"  WARNING: no pin for {src.name}; falling back to "
                          f"auto-detection (result may be unreliable)", file=sys.stderr)
                tier, axis, per = "auto", ax, None
            if want_tiers and tier not in want_tiers:
                continue
            key = str(src)
            meta[key] = (str(src), axis, per)
            groups.setdefault(key, []).append((str(dst), tier))

        jobs = [(*meta[k], v) for k, v in groups.items()]
        n_jobs = a.jobs if a.jobs > 0 else max(1, min(12, (os.cpu_count() or 2) - 2))
        n_jobs = min(n_jobs, len(jobs)) or 1

        if n_jobs > 1 and len(jobs) > 1:
            print(f"analysing {len(jobs)} inputs / "
                  f"{sum(len(v) for v in groups.values())} outputs "
                  f"on {n_jobs} workers", file=sys.stderr)
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futs = {ex.submit(_analyse_group, j): j[0] for j in jobs}
                done = 0
                for fut in as_completed(futs):
                    rows.extend(fut.result())
                    done += 1
                    print(f"  [{done}/{len(jobs)}] {Path(futs[fut]).name}",
                          file=sys.stderr)
        else:
            for j in jobs:
                rows.extend(_analyse_group(j))
    elif a.input and a.output:
        pin = pins.get(Path(a.input).name, {})
        r = evaluate(Path(a.input), Path(a.output),
                     a.axis or pin.get("axis"),
                     a.period if a.period else pin.get("period_px"),
                     pin.get("tier", "auto"))
        r["output"] = Path(a.output).name
        rows.append(r)
    else:
        ap.error("give --input/--output or --suite")

    base = {}
    if a.baseline and Path(a.baseline).exists():
        for r in csv.DictReader(open(a.baseline)):
            base[r.get("output", r.get("image", ""))] = r

    hdr = (f"{'output':<44}{'tier':<11}{'ax':<3}{'period':>8}{'supp%':>7}{'margin':>8}"
           f"{'ctrl%':>7}{'signed':>8}{'over%':>7}{'contr':>7}{'gamma':>7}")
    if base:
        hdr += f"{'d-margin':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r.get('output', r['image'])[:51]:<52} {r['error']}")
            continue
        line = (f"{r['output'][:43]:<44}{r.get('tier','auto'):<11}{r['axis'][0]:<3}"
                f"{r['period_px']:>8.2f}{r['supp_pct']:>7.1f}{r['margin']:>8.1f}"
                f"{r['ctrl_pct']:>7.1f}{r.get('signed',0):>+8.3f}{r.get('over_pct',0):>7.1f}"
                f"{r.get('contrast',0):>7.3f}{r.get('gamma',0):>7.3f}")
        if base:
            b = base.get(r["output"])
            if b and b.get("margin"):
                d = r["margin"] - float(b["margin"])
                line += f"{d:>+10.1f}"
            else:
                line += f"{'-':>10}"
        print(line)

    ok = [r for r in rows if "error" not in r]
    if ok:
        print()
        # Iterate over tiers actually present, not a fixed list -- otherwise a
        # tier like "not-measurable" is measured, printed in the table, and then
        # silently dropped from every summary line.
        known = ["target", "guard", "unresolved", "auto"]
        present = [t for t in known if any(r.get("tier", "auto") == t for r in ok)]
        present += sorted({r.get("tier", "auto") for r in ok} - set(known))
        for t in present:
            sel = [r for r in ok if r.get("tier", "auto") == t]
            if not sel:
                continue
            m = [r["margin"] for r in sel]
            note = ("  (excluded from pass/fail)"
                    if t in ("unresolved", "not-measurable") else "")
            sg = [r.get("signed", 0.0) for r in sel]
            ct = [r.get("contrast", 1.0) for r in sel]
            print(f"{t:<11} {len(sel):>2} pairs   margin {np.mean(m):+6.1f}   "
                  f"signed {np.mean(sg):+6.3f}   contrast {np.mean(ct):5.3f}   "
                  f"below floor: {sum(1 for x in m if x <= 0)}{note}")

    if a.csv and ok:
        keys = ["image", "output", "tier", "axis", "period_px", "cycles", "amp_in",
                "amp_out", "supp_pct", "floor_pct", "margin", "ctrl_pct",
                "signed", "over_pct", "contrast", "gamma"]
        with open(a.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(ok)
        print(f"wrote {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
