from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from dataclasses import dataclass

from hybrid_infer_detail_preserving import parser as cli_parser

# GUI v0.5 deliberately exposes only the controls used by the public,
# user-facing workflows/tuning guidance in DOCUMENTATION.md.  The CLI still
# accepts every advanced parameter; hidden parameters retain parser defaults.
EXPOSED_DESTS = {
    # Restormer correction / runtime
    "band_axis", "device", "amp", "processing_size", "passes",
    "second_pass_strength", "luma_mode", "highlight_recovery_strength",
    # Local cleanup and tonal eligibility
    "flat_filter", "flat_luma_strength", "flat_chroma_strength", "flat_highpass", "flat_lowpass",
    # Residual profile
    "flat_profile", "flat_profile_luma_strength", "flat_profile_chroma_strength",
    "flat_profile_band_period",
    # Orthogonal residual-profile cleanup
    "orthogonal_profile", "orthogonal_profile_luma_strength", "orthogonal_profile_chroma_strength",
    # Documented local safety troubleshooting control
    "flat_local_edge_distance",
    # Broad/few-cycle dominant-surface cleanup
    "flat_surface_equalizer",
}

GROUP_ORDER = [
    "Restormer correction",
    "Flat-region cleanup",
    "Residual profile",
    "Broad residual cleanup",
]

SLIDERS = {
    "second_pass_strength", "highlight_recovery_strength",
    "flat_luma_strength", "flat_chroma_strength",
    "flat_profile_luma_strength", "flat_profile_chroma_strength",
    "orthogonal_profile_luma_strength", "orthogonal_profile_chroma_strength",
}


# Some CLI arguments intentionally accept arbitrary strings rather than declaring
# argparse choices.  The desktop GUI exposes the supported user-facing values
# explicitly so these become real combo boxes instead of free-form text fields.
CHOICE_OVERRIDES = {
    "device": ("auto", "cuda", "cpu", "mps"),
    # Keep the recommended luminance modes first and the legacy raw mode last.
    "luma_mode": ("directional", "directional-additive", "row", "raw"),
    # Orthogonal cleanup now has its own checkbox, so the GUI no longer needs
    # the legacy "both" band-axis shortcut. The CLI keeps it for compatibility.
    "band_axis": ("auto", "horizontal", "vertical"),
}

LABELS = {
    "device": "Device",
    "amp": "Use FP16 / AMP",
    "processing_size": "Processing size",
    "band_axis": "Band direction",
    "passes": "Passes",
    "second_pass_strength": "Second-pass strength",
    "luma_mode": "Luminance mode",
    "highlight_recovery_strength": "Highlight recovery",
    "flat_filter": "Enable flat-region cleanup",
    "flat_luma_strength": "Flat luminance strength",
    "flat_chroma_strength": "Flat chroma strength",
    "flat_highpass": "Shadow cutoff",
    "flat_lowpass": "Highlight cutoff",
    "flat_profile": "Enable residual profile",
    "flat_profile_luma_strength": "Profile luminance strength",
    "flat_profile_chroma_strength": "Profile chroma strength",
    "flat_profile_band_period": "Profile band period (px)",
    "orthogonal_profile": "Enable orthogonal cleanup",
    "orthogonal_profile_luma_strength": "Orthogonal luminance strength",
    "orthogonal_profile_chroma_strength": "Orthogonal chroma strength",
    "flat_local_edge_distance": "Object-edge protection distance",
    "flat_surface_equalizer": "Enable large-surface equalizer",
}


@dataclass(frozen=True)
class SettingSpec:
    dest: str
    option: str
    label: str
    default: object
    value_type: type | None
    choices: tuple | None
    help: str
    group: str
    slider: bool


def _group(dest: str) -> str:
    if dest in {"band_axis", "device", "amp", "processing_size", "passes", "second_pass_strength", "luma_mode", "highlight_recovery_strength"}:
        return "Restormer correction"
    if dest in {"flat_filter", "flat_luma_strength", "flat_chroma_strength", "flat_highpass", "flat_lowpass", "flat_local_edge_distance"}:
        return "Flat-region cleanup"
    if dest.startswith("flat_profile") or dest.startswith("orthogonal_"):
        return "Residual profile"
    if dest == "flat_surface_equalizer":
        return "Broad residual cleanup"
    return "Restormer correction"


# Explicit GUI ordering.  Do not rely on argparse declaration order or
# alphabetical dest sorting: this is the presentation order requested for the
# desktop application.
ITEM_ORDER = {
    # Restormer correction
    "band_axis": 0,
    "device": 1,
    "amp": 2,
    "processing_size": 3,
    "passes": 4,
    "second_pass_strength": 5,
    "luma_mode": 6,
    "highlight_recovery_strength": 7,
    # Flat-region cleanup
    "flat_filter": 100,
    "flat_luma_strength": 101,
    "flat_chroma_strength": 102,
    "flat_highpass": 103,
    "flat_lowpass": 104,
    "flat_local_edge_distance": 105,
    # Residual profile + its optional perpendicular pass
    "flat_profile": 200,
    "flat_profile_luma_strength": 201,
    "flat_profile_chroma_strength": 202,
    "flat_profile_band_period": 203,
    "orthogonal_profile": 204,
    "orthogonal_profile_luma_strength": 205,
    "orthogonal_profile_chroma_strength": 206,
    # Broad residual cleanup
    "flat_surface_equalizer": 300,
}


def _label(dest: str) -> str:
    return LABELS.get(dest, dest.replace("_", " ").replace("luma", "luminance").title())


HEX_RE = re.compile(r"^[0-9a-fA-F]{6}$")


def canonical_cutoff(value, fallback: str) -> str:
    """Return canonical #RRGGBB, using fallback for invalid GUI text."""
    text = str(value or "").strip()
    if text.startswith("#"):
        text = text[1:]
    if not HEX_RE.fullmatch(text):
        return fallback
    return "#" + text.upper()


def cutoff_luma(value: str) -> float:
    text = canonical_cutoff(value, "#000000")[1:]
    r, g, b = (int(text[i:i+2], 16) for i in (0, 2, 4))
    return 0.299 * r + 0.587 * g + 0.114 * b


def normalize_gui_processing_values(settings: dict) -> dict:
    """Sanitize GUI-editable values before they reach inference/export."""
    out = dict(settings)
    if "processing_size" in out:
        try:
            v = int(out["processing_size"])
        except Exception:
            v = 512
        v = max(256, min(2048, v))
        v = max(256, min(2048, ((v + 4) // 8) * 8))
        out["processing_size"] = v
    if "flat_local_edge_distance" in out:
        try:
            v = int(out["flat_local_edge_distance"])
        except Exception:
            v = 6
        # Negative guard distances have no distinct meaning in the algorithm:
        # <=0 already disables the guard, so expose 0 as the minimum.
        out["flat_local_edge_distance"] = max(0, min(100, v))
    if "flat_profile_band_period" in out:
        try:
            v = float(out["flat_profile_band_period"])
        except Exception:
            v = 0.0
        out["flat_profile_band_period"] = 0.0 if v <= 0 else max(1.0, min(7680.0, v))
    numeric_ranges = {
        "second_pass_strength": (0.0, 2.0),
        "highlight_recovery_strength": (0.0, 1.0),
        "flat_luma_strength": (0.0, 2.0),
        "flat_chroma_strength": (0.0, 2.0),
        "flat_profile_luma_strength": (0.0, 4.0),
        "flat_profile_chroma_strength": (0.0, 4.0),
        "orthogonal_profile_luma_strength": (-1.0, 4.0),
        "orthogonal_profile_chroma_strength": (-1.0, 4.0),
    }
    for key, (low, high) in numeric_ranges.items():
        if key in out:
            try:
                out[key] = max(low, min(high, float(out[key])))
            except Exception:
                pass

    shadow = canonical_cutoff(out.get("flat_highpass", "#232323"), "#000000")
    highlight = canonical_cutoff(out.get("flat_lowpass", "#efefef"), "#FFFFFF")
    if cutoff_luma(shadow) > cutoff_luma(highlight):
        # Without UI edit-order context, use the two safe endpoints. The live
        # GUI applies the more specific rule to only the field the user edited.
        shadow = "#000000"
        highlight = "#FFFFFF"
    out["flat_highpass"] = shadow
    out["flat_lowpass"] = highlight
    return out


PLUMBING_DESTS = {"help", "input", "output", "luma_model", "chroma_model", "debug_dir", "overwrite"}


def _all_parser_defaults() -> OrderedDict[str, object]:
    """Return inference defaults for visible and GUI-hidden processing options."""
    result: OrderedDict[str, object] = OrderedDict()
    for action in cli_parser()._actions:
        if action.dest in PLUMBING_DESTS:
            continue
        result[action.dest] = action.default
    return result


def specs() -> list[SettingSpec]:
    out: list[SettingSpec] = []
    for action in cli_parser()._actions:
        if action.dest not in EXPOSED_DESTS:
            continue
        value_type = getattr(action, "type", None)
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)) or isinstance(action.default, bool):
            value_type = bool
        opts = [o for o in action.option_strings if o.startswith("--")]
        option = opts[0] if opts else action.dest
        out.append(SettingSpec(
            action.dest,
            option,
            _label(action.dest),
            action.default,
            value_type,
            CHOICE_OVERRIDES.get(action.dest, tuple(action.choices) if action.choices is not None else None),
            action.help or "",
            _group(action.dest),
            action.dest in SLIDERS,
        ))
    rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    return sorted(out, key=lambda s: (rank.get(s.group, 999), ITEM_ORDER.get(s.dest, 1000), s.dest))


def default_settings() -> OrderedDict[str, object]:
    """Per-image GUI state contains only controls visible in the right panel."""
    defaults = _all_parser_defaults()
    return OrderedDict((s.dest, defaults[s.dest]) for s in specs())


def namespace(settings: dict) -> argparse.Namespace:
    """Build a complete inference namespace.

    Advanced parameters hidden from the GUI are filled from the authoritative
    CLI parser defaults, so reducing the panel does not change restoration
    behavior.
    """
    values = dict(_all_parser_defaults())
    values.update(normalize_gui_processing_values(settings))
    # The GUI exposes the base local strengths directly. Keep the legacy CLI
    # multiplier neutral so settings copied from the short-lived multiplier GUI
    # cannot silently scale the values shown to the user.
    values["flat_cleanup_strength"] = 1.0
    # Exposure locking is intentionally not user-editable in the desktop GUI.
    # Force it after merging settings so pasted settings from older GUI builds
    # cannot silently restore the former pass2 value.
    values["exposure_lock"] = "all"
    return argparse.Namespace(**values)
