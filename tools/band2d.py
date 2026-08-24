#!/usr/bin/env python3
"""band2d.py — measure the band with a 2D spectrum instead of a row profile.

WHY THIS EXISTS
---------------
Every 1-D row-profile statistic in this toolset mixes the band with scene
structure, because a row profile collapses the x axis and cannot tell a
coherent horizontal band from shirt folds, hair, or a table edge that happen to
have similar vertical frequencies.

On pwm1 that produced four wrong conclusions in a row:

    "SNR 1.07, no band, out of scope"      -> actually 91.7x prominence
    "phase incoherent across columns"      -> peak sits exactly on fx=0
    "a single tone explains only 21%"      -> true of the profile, not the band
    "the tool removes essentially none"    -> it removes 46.1%

A true rolling-shutter band is constant along x, so in the 2D spectrum it is a
peak on the fx = 0 axis at fy = 1/period. Scene structure is not confined to
that axis. Measuring the peak height there, against the local noise floor in
the same fy range, separates the two cleanly.

    prominence = F[fy = 1/P, fx ~ 0] / median(F over the same fy band, all fx)

WHAT THE NUMBERS MEAN
    prominence  > 20   strong band, well worth correcting
                5 - 20 real but modest
                < 5    little or nothing at that period
    reduction   drop in the peak between input and output, at the SAME period

USAGE
    python tools/band2d.py --input src.jpg --output dst.png --axis horizontal
    python tools/band2d.py --input src.jpg --axis horizontal          # input only
    python tools/band2d.py --suite img/tests --outdir img/tests_new/restored \\
        --pins tools/pins.json

The period is found from the 2D spectrum itself, so a wrong pin cannot mislead
it -- pass --period only to force a specific value.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

Image.MAX_IMAGE_PIXELS = None


# Log-channel weights. Scene structure is multiplicative and affects all three
# channels in proportion, so it largely cancels in a ratio; a flicker source
# with a different spectrum from the ambient does not. On pwm1 the band's
# prominence over the local noise floor is 105 in luma and 706 in logB-logR,
# and luma reports the 30 px harmonic while every ratio finds the 58.7 px
# fundamental.
#
# NOTE: this is a MEASUREMENT aid only. Driving a correction from the ratio was
# tried and failed -- see chroma_band.py. Use it to find the period and judge
# whether a band exists, not to fit one.
RATIOS = {
    "luma": None,
    "logB-logR": (-1.0, 0.0, 1.0),
    "logB-logG": (0.0, -1.0, 1.0),
    "logG-logR": (-1.0, 1.0, 0.0),
    "logB-(R+G)/2": (-0.5, -0.5, 1.0),
    "logB+logG-2logR": (-2.0, 1.0, 1.0),
}


def log_field(path, axis, ratio="luma", max_side=2000):
    """Log luma, or a log-channel ratio that suppresses scene structure."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    f = 1.0
    if max(w, h) > max_side:
        f = max_side / max(w, h)
        im = im.resize((int(w * f), int(h * f)), Image.LANCZOS)
    a = np.asarray(im, np.float64) / 255.0
    wts = RATIOS.get(ratio)
    if wts is None:
        y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        l = np.log(y + 0.02)
    else:
        la = np.log(a + 0.02)
        l = sum(c * la[..., i] for i, c in enumerate(wts))
    return (l if axis == "horizontal" else l.T), f


def log_luma(path, axis, max_side=2000):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    f = 1.0
    if max(w, h) > max_side:
        f = max_side / max(w, h)
        im = im.resize((int(w * f), int(h * f)), Image.LANCZOS)
    a = np.asarray(im, np.float64) / 255.0
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    l = np.log(y + 0.02)
    return (l if axis == "horizontal" else l.T), f


def spectrum(L, detrend_sigma=25.0):
    r = L - gaussian_filter(L, detrend_sigma)
    h, w = r.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    return np.abs(np.fft.rfft2(r * win)), h, w


def band_peak(F, h, period, fx_width=3):
    """Peak height on the fx ~ 0 axis at fy = 1/period."""
    i = int(round(h / period))
    if i < 2 or i >= F.shape[0] // 2:
        return 0.0
    return float(F[max(0, i - 1):i + 2, 0:fx_width].max())


def noise_floor(F, h, pmin=25.0, pmax=200.0):
    fy = np.fft.fftfreq(h)[: F.shape[0]]
    m = (np.abs(fy) > 1.0 / pmax) & (np.abs(fy) < 1.0 / pmin)
    if not m.any():
        return 1e-9
    return float(np.median(F[m, :]))


def find_period(L, pmin=20.0, pmax=200.0):
    """Best period from the fx ~ 0 axis of the 2D spectrum."""
    F, h, w = spectrum(L)
    fy = np.fft.fftfreq(h)[: F.shape[0]]
    axis = F[:, 0:3].max(axis=1)
    best, bp = None, 0.0
    for i in range(2, len(axis)):
        if fy[i] <= 0:
            continue
        p = 1.0 / fy[i]
        if not (pmin <= p <= pmax):
            continue
        if axis[i] > bp:
            best, bp = p, axis[i]
    return best


def analyse(src, dst=None, axis="horizontal", period=None, pmin=20.0, pmax=200.0,
            ratio="luma"):
    L, f = log_field(src, axis, ratio)
    p = period * f if period else find_period(L, pmin, pmax)
    if p is None:
        return None
    F, h, w = spectrum(L)
    b0 = band_peak(F, h, p)
    nf = noise_floor(F, h, pmin, pmax)
    out = {"period_px": p / f, "prominence": b0 / max(nf, 1e-9), "peak_in": b0}
    if dst is not None:
        Ld, _ = log_field(dst, axis, ratio)
        if Ld.shape == L.shape:
            Fd, hd, _ = spectrum(Ld)
            b1 = band_peak(Fd, hd, p)
            out["peak_out"] = b1
            out["reduction"] = 100.0 * (1 - b1 / max(b0, 1e-9))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--axis", choices=["horizontal", "vertical"], default="horizontal")
    ap.add_argument("--period", type=float, help="force a period (original px)")
    ap.add_argument("--pmin", type=float, default=20.0)
    ap.add_argument("--pmax", type=float, default=200.0)
    ap.add_argument("--suite", help="directory with inputs/")
    ap.add_argument("--outdir", help="directory of restored outputs")
    ap.add_argument("--pins", help="pins.json, for the band axis")
    ap.add_argument("--ratio", choices=sorted(RATIOS), default="luma",
                    help="field to measure in. Ratios suppress scene structure and often "
                         "expose the fundamental where luma shows only a harmonic.")
    ap.add_argument("--scan-ratios", action="store_true",
                    help="try every ratio, each with its own period search, and report "
                         "which sees the band most clearly.")
    ap.add_argument("--verify", action="store_true",
                    help="re-run over several search windows. A period that changes when "
                         "the window changes is a harmonic-family artefact, not a "
                         "measurement -- do not pin it.")
    a = ap.parse_args()

    hdr = f"{'image':<46}{'ax':<3}{'period':>9}{'promin':>9}{'reduction':>11}  verdict"
    print(hdr)
    print("-" * len(hdr))

    def emit(label, axis, r):
        if r is None:
            print(f"{label[:45]:<46}{axis[0]:<3}{'-':>9}{'-':>9}{'-':>11}  no period")
            return
        pr = r["prominence"]
        v = ("STRONG band" if pr > 20 else
             "modest band" if pr > 5 else "little or nothing")
        red = f"{r['reduction']:>10.1f}%" if "reduction" in r else f"{'-':>11}"
        print(f"{label[:45]:<46}{axis[0]:<3}{r['period_px']:>9.2f}{pr:>9.1f}{red}  {v}")

    if a.suite:
        pins = {}
        if a.pins:
            pins = json.load(open(a.pins)).get("pins", {})
        ins = Path(a.suite) / "inputs"
        outs = Path(a.outdir) if a.outdir else None
        seen = set()
        for src in sorted(ins.iterdir()):
            if not src.is_file():
                continue
            pin = pins.get(src.name, {})
            axis = pin.get("axis", "horizontal")
            dst = None
            if outs and outs.is_dir():
                cands = [p for p in outs.glob(src.stem + "_restored*.png")]
                if cands:
                    dst = sorted(cands)[-1]
            if src.name in seen:
                continue
            seen.add(src.name)
            try:
                r = analyse(src, dst, axis, a.period, a.pmin, a.pmax, a.ratio)
            except Exception as e:                       # noqa: BLE001
                print(f"{src.name[:45]:<46}  ERROR {e!r}")
                continue
            emit(src.name + (f"  vs {dst.name[:18]}" if dst else ""), axis, r)
        return 0

    if a.scan_ratios:
        if not a.input:
            ap.error("--scan-ratios needs --input")
        print()
        print(f"  {'ratio':<18}{'period':>10}{'prominence':>12}{'reduction':>12}")
        print("  " + "-" * 50)
        rows = []
        for name in sorted(RATIOS):
            try:
                r = analyse(Path(a.input), Path(a.output) if a.output else None,
                            a.axis, None, a.pmin, a.pmax, name)
            except Exception as e:                       # noqa: BLE001
                print(f"  {name:<18}ERROR {e!r}")
                continue
            if r is None:
                print(f"  {name:<18}{'no period':>10}")
                continue
            red = f"{r['reduction']:11.1f}%" if "reduction" in r else f"{'-':>12}"
            print(f"  {name:<18}{r['period_px']:>10.2f}{r['prominence']:>12.1f}{red}")
            rows.append((r["prominence"], name, r["period_px"]))
        if rows:
            rows.sort(reverse=True)
            pr, name, per = rows[0]
            print(f"\n  clearest: {name} at {per:.2f} px (prominence {pr:.1f})")
            pers = sorted(r[2] for r in rows if r[1] != "luma")
            if pers and pers[-1] / max(pers[0], 1e-9) > 1.2:
                print("  WARNING: ratios disagree on the period -- treat it as undetermined.")
        return 0

    if a.verify:
        if not a.input:
            ap.error("--verify needs --input")
        print()
        wins = [(a.pmin, a.pmax), (a.pmin, a.pmax / 2), (a.pmin, a.pmax / 4),
                (a.pmin * 2, a.pmax), (a.pmin, a.pmax * 2)]
        seen = []
        for lo, hi in wins:
            if hi <= lo * 2:
                continue
            try:
                r = analyse(Path(a.input), Path(a.output) if a.output else None,
                            a.axis, None, lo, hi, a.ratio)
            except Exception as e:                       # noqa: BLE001
                print(f"  window {lo:6.0f}-{hi:<6.0f}  ERROR {e!r}")
                continue
            if r is None:
                print(f"  window {lo:6.0f}-{hi:<6.0f}  no period")
                continue
            seen.append(r["period_px"])
            red = f"{r['reduction']:6.1f}%" if "reduction" in r else "     -"
            print(f"  window {lo:6.0f}-{hi:<6.0f}  P={r['period_px']:8.2f}  "
                  f"prominence {r['prominence']:8.1f}  reduction {red}")
        if len(seen) >= 2:
            seen = np.array(seen)
            base = float(np.median(seen))
            ratios = seen / base
            spread = float(ratios.max() / ratios.min())
            near_harm = all(min(abs(x - round(x)), abs(1 / x - round(1 / x))) < 0.12
                            for x in ratios)
            print()
            if spread < 1.05:
                print(f"  STABLE: period {base:.2f} across all windows -> safe to pin")
            elif near_harm:
                print(f"  HARMONIC FAMILY: answers are integer multiples of each other "
                      f"(spread {spread:.2f}x).\n  The smallest is usually the fundamental. "
                      f"Verify with a narrow window before pinning.")
            else:
                print(f"  UNSTABLE: spread {spread:.2f}x and not a harmonic family.\n"
                      f"  The period is not well determined -- do not pin it.")
        return 0

    if not a.input:
        ap.error("give --input (and optionally --output), or --suite")
    r = analyse(Path(a.input), Path(a.output) if a.output else None,
                a.axis, a.period, a.pmin, a.pmax, a.ratio)
    emit(Path(a.input).name, a.axis, r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
