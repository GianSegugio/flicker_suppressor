#!/usr/bin/env python3
"""autosettings.py — estimate band settings for a new image before processing.

WHAT IT DECIDES, AND WHY ONLY THESE
-----------------------------------
Estimated (evidence-backed, mechanical):

  band_axis                which way the bands run
  flat_profile             whether the profile stage should run at all
  flat_profile_mode        pwm / smooth, or hand off to the surface equalizer
  flat_profile_band_period the fundamental, with ranked alternatives

Those four account for most of the difference between the worst and best
results on the reference set. 0-q9gicytxxn4h1 went from margin -51.4 to +20.8
purely by switching flat_profile from false to pwm; every large failure during
development was a period error (62.2 / 59.56 / 110.33 / 30.02 / 112 all wrong on
one image).

KNOWN LIMITS
  * Greyscale images are capped at medium confidence -- there is no second
    opinion to check luma against.
  * luma is kept in the estimate despite being the weakest field: dropping it
    sent 0-q9gicytxxn4h1 from 148.52 px to 594.07 px (exactly 4x, reported as
    high confidence). No single field is trustworthy on its own.
  * The unanimity rule on colour ratios is deliberately strict. A weighted
    majority was considered and rejected: on fXidK4 the two dissenting ratios
    carry the HIGHEST prominence (278.1 and 112.6) yet point at 186.67 px,
    where the tool's own output shows -4.7% reduction, while the majority at
    80.00 px shows +39.1%. A prominence-weighted majority would have chosen the
    wrong period. Strict agreement demotes some correct images to low, which
    costs a manual entry but is never wrong.

NOT estimated, deliberately:

  strengths, pass counts, guard thresholds

Every one of those was settled by eye against the metric, and the metric lost:
singerpwm v9 (+8.3) was chosen over v11 (+11.0); polish strength 0.33 beat 0.7
and 1.2; 0-q9gicytxxn4h1 gained 0.4 margin and blew out the lamp cores. There is
no measurement here that predicts those calls, so this leaves them at defaults.

HOW THE PERIOD IS FOUND
-----------------------
Two lessons drive the method.

1. Measure on a log-channel RATIO, not luma. Scene structure is multiplicative
   and cancels in a ratio; a flicker source with a different spectrum does not.
   On pwm1 the band's prominence is 105 in luma and 706 in logB-logR -- and luma
   reports the 30 px harmonic while every ratio finds the 58.7 px fundamental.

2. Search several windows and several ratios, and only trust an answer they
   agree on. A period that changes with the search window is a harmonic-family
   artefact. 3136031 reads 112 / 70 / 46.67 depending on the window, and its
   measured reduction flips between +21.4% and -27.5% across those -- so it is
   reported as undetermined rather than pinned.

CONFIDENCE
  high    ratios agree, windows agree, prominence >= 20
  medium  a clear winner but some disagreement -- alternatives offered
  low     weak or contradictory evidence; caller should leave defaults and warn

USAGE
    python autosettings.py --input photo.jpg --json photo.settings.json
    python autosettings.py --suite img/tests --csv estimates.csv
    python autosettings.py --input photo.jpg --base my_defaults.json --json out.json

The JSON is a complete recipe when --base is given (defaults merged with the
estimate), otherwise just the estimated keys plus a diagnostics block.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _gauss_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * float(sigma))))
    k = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (k / max(sigma, 1e-6)) ** 2)
    return k / k.sum()


def gaussian_filter(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur, numpy only.

    scipy.ndimage would do this in one call, but scipy is not a dependency of
    the shipped application and bundling it through Nuitka for two blurs is not
    worth ~100 MB and the extension-module fragility. Edges use reflect padding,
    matching scipy's mode="reflect" closely enough for a detrend.
    """
    k = _gauss_kernel(sigma)
    r = len(k) // 2
    out = np.pad(a.astype(np.float64), ((r, r), (0, 0)), mode="reflect")
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, out)
    out = np.pad(out, ((0, 0), (r, r)), mode="reflect")
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, out)

Image.MAX_IMAGE_PIXELS = None

RATIOS = {
    "luma": None,
    "logB-logR": (-1.0, 0.0, 1.0),
    "logG-logR": (-1.0, 1.0, 0.0),
    "logB-logG": (0.0, -1.0, 1.0),
    "logB-(R+G)/2": (-0.5, -0.5, 1.0),
}
WINDOWS = [(8.0, 200.0), (8.0, 100.0), (8.0, 400.0), (16.0, 200.0)]
ANALYSIS_SIDE = 1400


def is_greyscale(a, tol=0.02):
    """Ratios carry no information when the channels are identical."""
    return bool(max(float(np.abs(a[..., 0] - a[..., 1]).mean()),
                    float(np.abs(a[..., 1] - a[..., 2]).mean())) < tol)


def load_fields(path, max_side=ANALYSIS_SIDE):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    f = 1.0
    if max(w, h) > max_side:
        f = max_side / max(w, h)
        im = im.resize((max(1, int(w * f)), max(1, int(h * f))), Image.LANCZOS)
    a = np.asarray(im, np.float64) / 255.0
    la = np.log(a + 0.02)
    grey = is_greyscale(a)
    out = {}
    for name, wts in RATIOS.items():
        if wts is None:
            y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
            out[name] = np.log(y + 0.02)
        elif not grey:
            out[name] = sum(c * la[..., i] for i, c in enumerate(wts))
    return out, f, im.size, grey


def spectrum(L):
    r = L - gaussian_filter(L, 25)
    h, w = r.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    return np.abs(np.fft.rfft2(r * win)), h


def peak_and_floor(L, pmin, pmax):
    """Best fx~0 peak in the period window, and the local noise floor."""
    F, h = spectrum(L)
    fy = np.fft.fftfreq(h)[: F.shape[0]]
    axis = F[:, 0:3].max(axis=1)
    best_p, best_v = None, 0.0
    for i in range(2, len(axis)):
        if fy[i] <= 0:
            continue
        p = 1.0 / fy[i]
        if pmin <= p <= pmax and axis[i] > best_v:
            best_p, best_v = p, axis[i]
    m = (np.abs(fy) > 1.0 / pmax) & (np.abs(fy) < 1.0 / pmin)
    floor = float(np.median(F[m, :])) if m.any() else 1e-9
    return best_p, best_v, max(floor, 1e-9)


def scan(fields, axis, scale):
    """Every field x every window. Returns rows in ORIGINAL pixel units.

    luma is INCLUDED, though it is often the weakest field. Excluding it was
    tried and rejected. It is wrong whenever it disagrees with a ratio
    consensus -- my_photo_input (589.57 vs 687.83), xdn15oqgxnya1 (103.56 vs
    49.05), dGhU9 (prominence 30.6 vs 470.3), j7PMB0W9 (140.00 vs 62.22),
    ge107vg31 (405.01 vs 48-87) -- but on 0-q9gicytxxn4h1 it was the field
    holding the estimate at the true 148.5 px; without it the ratios locked
    onto 594.07, exactly 4x the fundamental, and reported high confidence.

    No single field is reliable. That is what the ratio-agreement check exists
    to detect, and it is the only defence that has not backfired.
    """
    rows = []
    for name, L in fields.items():
        Z = L if axis == "horizontal" else L.T
        for pmin, pmax in WINDOWS:
            p, v, fl = peak_and_floor(Z, pmin, pmax)
            if p is None:
                continue
            rows.append({"ratio": name, "window": [pmin, pmax],
                         "period": p / scale, "prominence": v / fl})
    return rows


def ratio_agreement(rows, tol=1.2):
    """Do the colour ratios independently land on the same period?

    The cluster score alone cannot see this: it sums prominence across ratios
    and windows, so a strong minority answer is absorbed rather than flagged.
    band2d --scan-ratios does make the comparison, and it caught j7PMB0W9...
    where four ratios say 62.22 px, logG-logR says 112.00 and luma says 140.00
    (112/62.22 = 1.80, not a harmonic). The estimator rated that "high".

    luma is excluded from the vote. It disagreed with the ratios on dGhU9
    (125.00 vs prominence 30.6), xdn15oqgxnya1 (103.56 vs 49.05, roughly H2)
    and j7PMB0W9 -- it is the least reliable field and should not get a say.

    Returns (spread, best_period_per_ratio). spread <= tol means agreement.
    """
    best = {}
    for r in rows:
        if r["ratio"] == "luma":
            continue
        cur = best.get(r["ratio"])
        if cur is None or r["prominence"] > cur["prominence"]:
            best[r["ratio"]] = r
    pers = sorted(v["period"] for v in best.values())
    if len(pers) < 2:
        return 1.0, {k: round(v["period"], 2) for k, v in best.items()}
    spread = pers[-1] / max(pers[0], 1e-9)
    if harmonic_of(pers[0], pers[-1]):        # same answer seen at two scales
        spread = 1.0
    return spread, {k: round(v["period"], 2) for k, v in best.items()}


def consensus(rows, tol=0.12):
    """Cluster periods; return clusters sorted by summed prominence."""
    if not rows:
        return []
    clusters = []
    for r in sorted(rows, key=lambda r: -r["prominence"]):
        for c in clusters:
            if abs(r["period"] / c["period"] - 1.0) <= tol:
                c["members"].append(r)
                c["score"] += r["prominence"]
                c["period"] = float(np.median([m["period"] for m in c["members"]]))
                break
        else:
            clusters.append({"period": r["period"], "score": r["prominence"],
                             "members": [r]})
    return sorted(clusters, key=lambda c: -c["score"])


def harmonic_of(a, b, tol=0.10):
    """True when a and b are related by a small integer ratio."""
    if a <= 0 or b <= 0:
        return False
    x = max(a, b) / min(a, b)
    return abs(x - round(x)) <= tol and 2 <= round(x) <= 6


def aspect_prior(size, portrait_ratio=1.10):
    """The tool's existing rule: portrait implies rotated sensor rows."""
    w, h = size
    return "vertical" if (h / max(w, 1)) >= portrait_ratio else "horizontal"


def estimate(path, base=None, axis=None):
    """Estimate the period on a GIVEN axis.

    The axis is NOT chosen here. Scoring both axes by band prominence was tried
    and disagreed with the reference set on 4 of 24 images (3136031, mGec5,
    qVyMq, ge107vg31), rating all four "high" -- scene structure elongated along
    one axis outscores a real band often enough to be unsafe. The tool's aspect
    prior is the better default: on jDOn4whY... the content statistic preferred
    vertical (H=0.040 V=0.062) and the prior's horizontal was correct.

    Pass `axis` explicitly to honour a manual override.
    """
    fields, scale, size, grey = load_fields(path)
    orig = (int(size[0] / scale), int(size[1] / scale))
    axis = axis or aspect_prior(orig)
    per_axis = {}
    for ax in ("horizontal", "vertical"):
        rows = scan(fields, ax, scale)
        cl = consensus(rows)
        per_axis[ax] = {"rows": rows, "clusters": cl,
                        "best": cl[0]["score"] if cl else 0.0}
    other = "vertical" if axis == "horizontal" else "horizontal"
    clusters = per_axis[axis]["clusters"]
    axis_margin = per_axis[axis]["best"] / max(per_axis[other]["best"], 1e-9)
    axis_note = ("" if axis_margin >= 0.7 else
                 f"the {other} axis scores {1/max(axis_margin,1e-9):.1f}x higher; "
                 f"if the bands do not run as expected, try switching the axis")

    diag = {
        "greyscale": grey,
        "axis_source": "prior/override",
        "axis_note": axis_note,
        "source_size": [int(size[0] / scale), int(size[1] / scale)],
        "axis_score": {k: round(per_axis[k]["best"], 1) for k in per_axis},
        "axis_margin": round(axis_margin, 2),
    }

    if not clusters:
        return {"confidence": "low", "reason": "no periodic band found on either axis",
                "band_axis": axis, "candidates": [], "settings": {}, "diagnostics": diag}

    spread, per_ratio = ratio_agreement(per_axis[axis]["rows"])
    diag["ratio_periods"] = per_ratio
    diag["ratio_spread"] = round(spread, 2)

    top = clusters[0]
    n_ratios = len({m["ratio"] for m in top["members"]})
    n_windows = len({tuple(m["window"]) for m in top["members"]})
    prom = float(np.median([m["prominence"] for m in top["members"]]))
    # size is the ANALYSIS size but periods are in original pixels, so the band
    # length has to be converted back or the cycle count is wrong by `scale`
    # (0-q9gicytxxn4h1 reported 6.3 cycles instead of 28).
    band_len = (size[1] if axis == "horizontal" else size[0]) / scale
    cycles = band_len / max(top["period"], 1e-9)

    # A rival cluster that is NOT a harmonic of the winner means two different
    # answers, not one answer seen at two scales.
    rival = None
    for c in clusters[1:]:
        if c["score"] > 0.45 * top["score"] and not harmonic_of(c["period"], top["period"]):
            rival = c
            break

    need_ratios = 1 if grey else 3
    need_ratios_med = 1 if grey else 2
    # A real rolling-shutter band is far stronger on its own axis. When the two
    # axes score close together the winner is usually scene structure, so the
    # axis is treated as undetermined rather than guessed.
    ratios_agree = grey or spread <= 1.2
    if (prom >= 20 and n_ratios >= need_ratios and n_windows >= 3
            and rival is None and ratios_agree and 3 <= cycles <= 80):
        # Greyscale offers no cross-check: only luma is available, and window
        # stability cannot detect that luma itself is the wrong instrument.
        # BW_my_photo_input is STABLE at 589.57 across every window while the
        # colour version of the same scene gives 687.83 from all five ratios.
        conf = "medium" if grey else "high"
        reason = ("greyscale: no colour ratio available to cross-check the period"
                  if grey else "")
    elif prom >= 8 and n_ratios >= need_ratios_med and rival is None and ratios_agree:
        conf = "medium"
        reason = "band found but agreement is partial; check the period"
    else:
        conf = "low"
        bits = []
        if prom < 8:
            bits.append(f"prominence {prom:.1f} is too low to be sure a band exists")
        if rival is not None:
            bits.append(f"a competing period of {rival['period']:.1f} px is not a "
                        f"harmonic of {top['period']:.1f} px")
        if not ratios_agree:
            bits.append(f"colour ratios disagree on the period by {spread:.2f}x "
                        f"({', '.join(f'{k}={v}' for k, v in sorted(per_ratio.items()))})")
        if not (3 <= cycles <= 80):
            bits.append(f"{cycles:.1f} cycles is outside the workable 3-80 range")
        reason = "; ".join(bits) or "evidence too weak"

    cands = []
    for c in clusters[:6]:
        cyc = band_len / max(c["period"], 1e-9)
        cands.append({
            "period_px": round(c["period"], 2),
            "cycles": round(cyc, 1),
            "score": round(c["score"], 1),
            "ratios": sorted({m["ratio"] for m in c["members"]}),
            "relation": ("primary" if c is top else
                         "harmonic" if harmonic_of(c["period"], top["period"]) else
                         "independent"),
        })

    settings = {}
    if conf != "low":
        settings = {
            "band_axis": axis,
            "flat_profile": True,
            "flat_profile_mode": "pwm",
            "flat_profile_band_period": round(top["period"], 3),
        }
        # Under ~3 cycles this is an illumination gradient, not a band; the
        # surface equalizer is the right stage. led-lights is that case: one
        # large irregular dark region, no period, and its equalizer-only recipe
        # takes the brightness spread from 186% to 58%.
        if cycles < 3.0:
            settings.update({"flat_profile": False, "flat_profile_mode": "smooth",
                             "flat_surface_equalizer": True})

    return {"confidence": conf, "reason": reason, "band_axis": axis,
            "period_px": round(top["period"], 3), "cycles": round(cycles, 1),
            "prominence": round(prom, 1), "candidates": cands,
            "axis_note": axis_note,
            "settings": ({**base, **settings} if base else settings),
            "diagnostics": diag}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input")
    ap.add_argument("--json", help="write the estimate here")
    ap.add_argument("--base", help="defaults JSON to merge the estimate into")
    ap.add_argument("--axis", choices=["horizontal", "vertical"],
                    help="override the band axis; default is the aspect prior")
    ap.add_argument("--suite", help="directory with inputs/ to batch-estimate")
    ap.add_argument("--csv")
    a = ap.parse_args()

    base = json.load(open(a.base)) if a.base else None

    if a.suite:
        ins = Path(a.suite) / "inputs"
        hdr = f"{'image':<46}{'conf':<8}{'ax':<3}{'period':>9}{'cyc':>7}{'prom':>8}{'cands':>7}"
        print(hdr); print("-" * len(hdr))
        rows = []
        for p in sorted(ins.iterdir()):
            if not p.is_file():
                continue
            try:
                r = estimate(p, base, a.axis)
            except Exception as e:                       # noqa: BLE001
                print(f"{p.name[:45]:<46}ERROR {e!r}")
                continue
            rows.append({"image": p.name, **{k: r.get(k) for k in
                        ("confidence", "band_axis", "period_px", "cycles", "prominence")},
                        "n_candidates": len(r["candidates"]), "reason": r["reason"]})
            print(f"{p.name[:45]:<46}{r['confidence']:<8}{r['band_axis'][0]:<3}"
                  f"{(r.get('period_px') or 0):>9.2f}{(r.get('cycles') or 0):>7.1f}"
                  f"{(r.get('prominence') or 0):>8.1f}{len(r['candidates']):>7}")
            if r["confidence"] == "low":
                print(f"{'':<46}  -> {r['reason']}")
        if a.csv and rows:
            import csv as _csv
            with open(a.csv, "w", newline="") as fh:
                wr = _csv.DictWriter(fh, fieldnames=list(rows[0]))
                wr.writeheader(); wr.writerows(rows)
            print(f"\nwrote {a.csv}")
        return 0

    if not a.input:
        ap.error("give --input or --suite")
    r = estimate(Path(a.input), base, a.axis)
    print(json.dumps(r, indent=2))
    if a.json:
        json.dump(r["settings"] if base else r, open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}", file=sys.stderr)
    return 0 if r["confidence"] != "low" else 2


if __name__ == "__main__":
    sys.exit(main())
