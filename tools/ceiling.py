#!/usr/bin/env python3
"""ceiling.py — is there a stationary periodic band to remove at all?

For each image, scan periods and report, at the best one:

  SNR      band-mode amplitude divided by the mean at four control frequencies
           (0.80x, 0.86x, 1.17x, 1.24x). Below ~1.3 there is less energy at the
           candidate period than at its neighbours, i.e. no band.
  ceiling  max achievable suppression = 100*(1 - 1/SNR). Negative means the
           "band" is weaker than the surrounding scene texture.
  fold%    what a phase-folded template removes from the 1-D profile. A folded
           template represents ANY waveform (sine, square, sawtooth, pulse), so
           a low number here rules out the whole family, not just one basis.
  drift    period fitted on the first half vs the second half. A large split
           means non-stationary -- the case a short-time model would target.

Motivated by getting 3136031 wrong: I called it a sawtooth from the shape of a
profile plot, then measured SNR 0.98 and a folded template removing 4%. The
ramp-then-drop was stage lighting, not flicker. This script is the check that
should come before any new waveform basis.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_filter1d

Image.MAX_IMAGE_PIXELS = None


def load(path, axis, max_side=1600):
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


def profile(L):
    n = L.shape[0]
    p = np.median(L, axis=1)
    return p - gaussian_filter1d(p, max(4.0, n * 0.04), mode="nearest")


def coherent(sig, P, nh=1):
    t = np.arange(len(sig))
    tot = 0.0
    for k in range(1, nh + 1):
        if P / k < 2.2:
            break
        tot += abs(np.sum(sig * np.exp(-2j * np.pi * k * t / P))) ** 2
    return tot


def best_period(sig, lo, hi, nh=1, npts=3000):
    ps = np.linspace(lo, hi, npts)
    v = np.array([coherent(sig, p, nh) for p in ps])
    return float(ps[int(np.argmax(v))])


def amp2d(L, P, harms=(1, 2, 3)):
    n, w = L.shape
    t = np.arange(n)[:, None]
    tot = 0
    for k in harms:
        if P / k < 2.5:
            break
        z = L * np.exp(-2j * np.pi * k * t / P)
        zr = gaussian_filter(z.real, (P * 1.2, P * 1.0))
        zi = gaussian_filter(z.imag, (P * 1.2, P * 1.0))
        tot = tot + (2 * np.abs(zr + 1j * zi)) ** 2
    a = np.sqrt(tot)
    return float(a[int(n * .06):int(n * .94), int(w * .06):int(w * .94)].mean())


def fold_remove(sig, P, nb=32):
    n = len(sig)
    ph = ((np.arange(n) % P) / P * nb).astype(int) % nb
    acc = np.zeros(nb); cnt = np.zeros(nb)
    np.add.at(acc, ph, sig); np.add.at(cnt, ph, 1.0)
    w = acc / np.maximum(cnt, 1.0); w -= w.mean()
    full = np.interp((np.arange(n) % P) / P * nb, np.arange(nb + 1), np.r_[w, w[0]])
    return 100.0 * (1 - (sig - full).std() / max(sig.std(), 1e-9)), w


CASES = [
    ("pwm1.jpg", "horizontal", 20, 120),
    ("3136031-ab94bbe259dba5c2babece4c95caa04f.jpg", "horizontal", 30, 160),
    ("fXidK4RrGlIdtvLYCjJv33chKl3ZnimcSx.jpg", "vertical", 30, 200),
    ("pwm_orange.jpg", "vertical", 6, 300),
    # controls: images known to work
    ("singerpwm.jpg", "vertical", 12, 40),
    ("test3_input.jpg", "horizontal", 20, 90),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="/home/claude/work/tests/inputs")
    a = ap.parse_args()

    hdr = (f"{'image':<40}{'P':>9}{'cyc':>6}{'SNR':>7}{'ceiling':>9}"
           f"{'fold%':>7}{'drift%':>8}  verdict")
    print(hdr)
    print("-" * len(hdr))
    from pathlib import Path
    for name, axis, lo, hi in CASES:
        p = Path(a.dir) / name
        if not p.exists():
            print(f"{name[:39]:<40}  missing")
            continue
        L, f = load(p, axis)
        sig = profile(L)
        P = best_period(sig, lo * f, hi * f)
        n = len(sig)
        s = amp2d(L, P)
        c = np.mean([amp2d(L, P * r) for r in (0.80, 0.86, 1.17, 1.24)])
        snr = s / max(c, 1e-9)
        ceil = 100.0 * (1 - 1.0 / max(snr, 1e-9))
        fold, _ = fold_remove(sig, P)
        h = len(sig) // 2
        p1 = best_period(sig[:h], lo * f, hi * f)
        p2 = best_period(sig[h:], lo * f, hi * f)
        drift = 100.0 * abs(p1 - p2) / max(P, 1e-9)
        if snr < 1.3:
            v = "NO BAND - out of scope"
        elif drift > 10:
            v = "non-stationary -> short-time model"
        else:
            v = "stationary band present"
        print(f"{name[:39]:<40}{P/f:>9.2f}{n/P:>6.1f}{snr:>7.2f}{ceil:>8.1f}%"
              f"{fold:>7.1f}{drift:>7.1f}%  {v}")
    print()
    print("SNR < 1.3 means there is less energy at the candidate period than at")
    print("neighbouring frequencies. No waveform basis can help; there is no band.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
