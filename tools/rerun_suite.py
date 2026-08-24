#!/usr/bin/env python3
"""rerun_suite.py — replay the regression suite with each image's own settings.

WHY THIS EXISTS
---------------
Every image in tests/restored was produced with DIFFERENT settings (band axis,
pass count, which cleanup stages were on, per-stage strengths). A plain batch
run like

    python hybrid_infer_detail_preserving.py --input tests/inputs --output out/

applies one flag set to all 24 images, so its outputs are NOT comparable to the
baseline you measured. Any margin change you saw would be "I changed the
settings", not "I changed the algorithm".

This script reads each *_restored*.json beside the existing outputs, converts it
back into CLI flags, and re-runs that exact recipe on the matching input. Verified
against the archive: all 119 keys in those JSONs map 1:1 onto CLI flags.

USAGE
-----
    # see the commands without running anything
    python tools/rerun_suite.py --suite img/tests --out img/tests/restored_new --dry-run

    # actually run
    python tools/rerun_suite.py --suite img/tests --out img/tests/restored_new

    # one image while iterating on a patch
    python tools/rerun_suite.py --suite img/tests --out img/tests/scratch \\
        --only jDOn4whYSCFqB6NK023zIouFqVB

Then:
    python tools/bandmetrics.py --suite img/tests --csv step1.csv --baseline baseline.csv

NOTE: bandmetrics.py --suite expects inputs/ and restored/ side by side. To
measure a fresh run, either point --suite at a directory laid out that way, or
temporarily swap restored/ for restored_new/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Flags declared with action="store_true": emit the flag when true, nothing when false.
STORE_TRUE = {
    "orthogonal_profile", "no_row_anchor", "flat_filter", "flat_profile",
    "flat_surface_equalizer", "flat_allow_mean_shift",
}
# Flags declared with argparse.BooleanOptionalAction: emit --x or --no-x.
BOOL_OPTIONAL = {
    "amp", "restormer", "flat_profile_adaptive", "flat_profile_no_harm",
    "flat_profile_pwm_polish",
}
# Colour values: strip the leading '#'. PowerShell treats '#' as a comment start,
# and the parser accepts hashless hex (see DOCUMENTATION.md section 8.4).
COLOR_KEYS = {"flat_highpass", "flat_lowpass"}

import re as _re
# Accept _restored, _restored_v0 ... _restored_vN so new recipe variants (v2, v3...)
# can be tested without overwriting the baseline outputs.
STEM_RE = _re.compile(r"^(?P<base>.+?)_restored(?:_v\d+)?$")


def flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def settings_to_args(cfg: dict) -> list[str]:
    out: list[str] = []
    for k, v in cfg.items():
        if k in STORE_TRUE:
            if bool(v):
                out.append(flag(k))
            continue
        if k in BOOL_OPTIONAL:
            out.append(flag(k) if bool(v) else "--no-" + k.replace("_", "-"))
            continue
        if k in COLOR_KEYS:
            out += [flag(k), str(v).lstrip("#")]
            continue
        if isinstance(v, bool):                      # unexpected bool: be loud
            raise SystemExit(f"unhandled boolean setting {k!r}; add it to the tables above")
        out += [flag(k), repr(v) if isinstance(v, float) and v != v else str(v)]
    return out


def find_jobs(suite: Path):
    ins, outs = suite / "inputs", suite / "restored"
    if not ins.is_dir() or not outs.is_dir():
        raise SystemExit(f"expected {ins} and {outs}")
    for j in sorted(outs.glob("*.json")):
        m = STEM_RE.match(j.stem)
        if not m:
            continue
        base = m.group("base")
        stem = j.stem
        cand = list(ins.glob(base + ".*"))
        if not cand:
            print(f"  skip {j.name}: no matching input", file=sys.stderr)
            continue
        yield base, cand[0], j, stem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, help="directory holding inputs/ and restored/")
    ap.add_argument("--out", required=True, help="destination directory for new outputs")
    ap.add_argument("--script", default="hybrid_infer_detail_preserving.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--luma-model", default="models/y.pth")
    ap.add_argument("--chroma-model", default="models/chroma.pth")
    ap.add_argument("--only", action="append", help="run only this base name (repeatable)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    suite = Path(a.suite)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = list(find_jobs(suite))
    if a.only:
        want = set(a.only)
        jobs = [j for j in jobs if j[0] in want]
    if not jobs:
        raise SystemExit("nothing to run")

    failures = []
    for i, (base, src, jpath, stem) in enumerate(jobs, 1):
        cfg = json.load(open(jpath))
        dst = out / (stem + ".png")
        cmd = [a.python, a.script, "--input", str(src), "--output", str(dst), "--overwrite"]
        # Model paths are deliberately excluded from the dev JSON, so supply them.
        # Harmless when the recipe has restormer=false: the script won't load them.
        if cfg.get("restormer", True):
            cmd += ["--luma-model", a.luma_model, "--chroma-model", a.chroma_model]
        cmd += settings_to_args(cfg)

        print(f"[{i}/{len(jobs)}] {base}")
        if a.dry_run:
            print("   " + " ".join(cmd))
            continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            failures.append(base)
            print(f"   FAILED rc={r.returncode}", file=sys.stderr)
        else:
            # Copy the recipe next to the new output so bandmetrics can read
            # band_axis from it exactly as it does for the baseline.
            shutil.copyfile(jpath, dst.with_suffix(".json"))

    if failures:
        print(f"\n{len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    if not a.dry_run:
        print(f"\nwrote {len(jobs)} outputs to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
