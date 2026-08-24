#!/usr/bin/env python3
"""make_variant.py — derive a new recipe JSON from an existing one.

Creates <base>_restored_v<N>.json in the same folder with the given settings
overridden, so a new recipe can be tested without touching the baseline outputs.

    # switch 0-q9gicytxxn4h1 to PWM mode as a new v2 recipe
    python tools/make_variant.py --suite img/tests \\
        --from 0-q9gicytxxn4h1_restored_v1 --to v2 \\
        --set flat_profile=true --set flat_profile_mode=pwm

    # same, plus a manual period and the polish on
    python tools/make_variant.py --suite img/tests \\
        --from test3_input_restored --to v2 \\
        --set flat_profile_mode=pwm --set flat_profile_pwm_polish=true

Then:
    python tools/rerun_suite.py --suite img/tests --out img/tests_new/restored --only 0-q9gicytxxn4h1
    python tools/bandmetrics.py --suite img/tests_new --pins tools/pins.json --tier target

Values are parsed as JSON when possible (true/false/numbers), else kept as text.
Unknown keys are rejected against the source recipe, so a typo fails loudly
instead of being silently ignored by the CLI parser.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, help="directory holding inputs/ and restored/")
    ap.add_argument("--from", dest="src", required=True,
                    help="source recipe stem, e.g. 0-q9gicytxxn4h1_restored_v1")
    ap.add_argument("--to", required=True, help="new variant tag, e.g. v2")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="setting to override (repeatable)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing variant")
    ap.add_argument("--add", action="append", default=[], metavar="KEY=VALUE",
                    help="add a setting NOT present in the source recipe (repeatable). "
                         "Needed for options introduced after the JSONs were exported, "
                         "e.g. tone_restore_strength. --set still rejects unknown keys "
                         "so typos stay loud.")
    a = ap.parse_args()

    restored = Path(a.suite) / "restored"
    src = restored / (a.src + ".json")
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        return 1

    cfg = json.load(open(src))

    for item, allow_new in [(i, False) for i in a.set] + [(i, True) for i in a.add]:
        flag = "--add" if allow_new else "--set"
        if "=" not in item:
            print(f"{flag} needs KEY=VALUE, got {item!r}", file=sys.stderr)
            return 1
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in cfg and not allow_new:
            print(f"unknown setting {k!r} (not present in {src.name}).", file=sys.stderr)
            print(f"If this option was added after the recipe was exported, use "
                  f"--add {k}={v.strip()}", file=sys.stderr)
            return 1
        new = parse_value(v.strip())
        if k in cfg:
            print(f"  {k}: {cfg[k]!r} -> {new!r}")
        else:
            print(f"  {k}: (new) -> {new!r}")
        cfg[k] = new

    m = re.match(r"^(?P<base>.+?)_restored(?:_v\d+)?$", a.src)
    if not m:
        print(f"could not parse a base name from {a.src!r}", file=sys.stderr)
        return 1
    tag = a.to if a.to.startswith("v") else "v" + a.to
    dst = restored / f"{m.group('base')}_restored_{tag}.json"
    if dst.exists() and not a.force:
        print(f"{dst} exists; pass --force to overwrite", file=sys.stderr)
        return 1

    dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"wrote {dst}")
    print(f"\nNOTE: this recipe has no baseline row, so bandmetrics will show '-' in\n"
          f"d-margin. Compare it by eye against {a.src}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
