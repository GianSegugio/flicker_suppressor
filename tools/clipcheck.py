#!/usr/bin/env python3
"""clipcheck.py — measure highlight clipping and gamut loss against the source.

Used to isolate which pipeline stage is blowing out the two stage lamps in
0-q9gicytxxn4h1. Each run prints one line, so a bisection reads as a table.

    python tools/clipcheck.py --input img/tests/inputs/0-q9gicytxxn4h1.jpg ^
        --output img/tests_new/restored/0-q9gicytxxn4h1_restored_v5.png --label v5

    # region around the left lamp only (much more sensitive than whole-frame)
    python tools/clipcheck.py --input ... --output ... --box 1300,3550,2050,4150

COLUMNS
  >0.99%   fraction of pixels with luminance above 0.99  (hard clipping)
  >0.95%   fraction above 0.95                            (near-clipping)
  ch>0.996 fraction with ANY RGB channel at the ceiling   (per-channel clip)
  p99      99th percentile luminance
  sat_lost mean saturation change on pixels that are bright in BOTH images;
           negative means the output desaturated them -- the signature of
           y_cbcr_to_rgb_preserve_y holding luminance and giving up chroma
  new_clip fraction of pixels clipped in the output that were NOT clipped in
           the source. This is the number that matters: it is damage, not
           inherited.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load(path, box=None):
    im = Image.open(path).convert("RGB")
    if box:
        im = im.crop(box)
    return np.asarray(im, np.float64) / 255.0


def luma(a):
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def sat(a):
    mx = a.max(-1)
    mn = a.min(-1)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--box", help="x0,y0,x1,y1 crop applied to both")
    ap.add_argument("--header", action="store_true", help="print the column header")
    a = ap.parse_args()

    box = tuple(int(v) for v in a.box.split(",")) if a.box else None
    src = load(a.input, box)
    dst = load(a.output, box)
    if src.shape != dst.shape:
        print(f"shape mismatch {src.shape} vs {dst.shape}", file=sys.stderr)
        return 1

    ys, yd = luma(src), luma(dst)
    hdr = (f"{'label':<22}{'>0.99%':>9}{'>0.95%':>9}{'ch>0.996%':>11}"
           f"{'p99':>8}{'sat_lost':>10}{'new_clip%':>11}")
    if a.header:
        print(hdr)
        print("-" * len(hdr))

    bright = (ys > 0.85) & (yd > 0.85)
    sat_lost = float((sat(dst)[bright] - sat(src)[bright]).mean()) if bright.any() else 0.0
    new_clip = float(((yd > 0.99) & (ys <= 0.99)).mean())

    print(f"{a.label[:21]:<22}"
          f"{100*(yd>0.99).mean():>9.3f}"
          f"{100*(yd>0.95).mean():>9.3f}"
          f"{100*(dst.max(-1)>0.996).mean():>11.3f}"
          f"{np.percentile(yd,99):>8.4f}"
          f"{sat_lost:>+10.4f}"
          f"{100*new_clip:>11.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
