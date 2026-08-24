#!/usr/bin/env python3
"""tonecurve.py — is the pipeline compressing highlights, or just shifting exposure?

Prints mean output/input luminance ratio by input level, measured on band-axis
envelopes so the flicker does not contaminate it.

    1.00 everywhere        -> unchanged
    uniform row (e.g. all 0.95) -> exposure shift, harmless
    FALLING left to right  -> highlight compression: shadows lifted, highlights
                              squeezed. This is what highlight recovery exists
                              to repair.

WHY THIS EXISTS
---------------
apply_highlight_recovery gates on `start=0.90`, so it only sees pixels whose
reference luminance exceeds 0.90. Measured on the test pairs, the compression
actually happens at 0.60-0.90, where almost nothing crosses that threshold:

    image                  bright px darkened   mean loss   gate catches
    test3_input                        100.0%      0.1247           0.0%
    0-q9gicytxxn4h1                    100.0%      0.1051           2.5%
    jDOn4whYSCFqB6NK...                 95.6%      0.0490           0.0%
    qVyMq                               89.9%      0.0453           0.0%
    singerpwm (--no-restormer)           0.1%      0.0010           0.0%

and the response curve confirms it is compression, not exposure:

    test3_input     1.275  1.122  0.959  0.887  0.828  0.799
                    (0.05)                             (0.90)

singerpwm, the only image with Restormer disabled, is flat -- the compression
comes from the neural passes.

USAGE
    python tools/tonecurve.py --input src.jpg --output dst.png --axis horizontal --period 38.15
    python tools/tonecurve.py --input src.jpg --output dst.png          # no band smoothing

Run it on a pre-P11 and a post-P11 output of the same image to see how much of
the compression the tone stage already removes.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

Image.MAX_IMAGE_PIXELS = None
BINS = [(0.05, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 0.60),
        (0.60, 0.75), (0.75, 0.90), (0.90, 1.01)]


def load(path, max_side):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    f = 1.0
    if max(w, h) > max_side:
        f = max_side / max(w, h)
        im = im.resize((int(w * f), int(h * f)), Image.LANCZOS)
    a = np.asarray(im, np.float64) / 255.0
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2], f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--axis", choices=["horizontal", "vertical"], default="horizontal")
    ap.add_argument("--period", type=float, default=0.0,
                    help="band period in ORIGINAL pixels; scaled internally")
    ap.add_argument("--label", default="")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--header", action="store_true")
    a = ap.parse_args()

    ry, f = load(a.input, a.max_side)
    py, _ = load(a.output, a.max_side)
    if ry.shape != py.shape:
        print(f"shape mismatch {ry.shape} vs {py.shape}", file=sys.stderr)
        return 1

    per = a.period * f
    ax = 0 if a.axis == "horizontal" else 1
    if per > 3.0:
        ry = gaussian_filter1d(ry, max(1.5, 0.6 * per), axis=ax, mode="nearest")
        py = gaussian_filter1d(py, max(1.5, 0.6 * per), axis=ax, mode="nearest")

    if a.header:
        h = f"{'label':<26}" + "".join(f"{f'{lo:.2f}-{hi:.2f}':>11}" for lo, hi in BINS)
        print(h)
        print("-" * len(h))

    row = f"{(a.label or 'ratio')[:25]:<26}"
    ratios = []
    for lo, hi in BINS:
        m = (ry >= lo) & (ry < hi)
        if m.sum() > 200:
            r = float(py[m].mean() / max(ry[m].mean(), 1e-9))
            ratios.append(r)
            row += f"{r:>11.3f}"
        else:
            row += f"{'-':>11}"
    print(row)

    if len(ratios) >= 3:
        span = ratios[0] - ratios[-1]
        verdict = ("COMPRESSION" if span > 0.08 else
                   "expansion" if span < -0.08 else "flat (exposure only)")
        print(f"{'':<26}shadow-to-highlight span {span:+.3f}  ->  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
