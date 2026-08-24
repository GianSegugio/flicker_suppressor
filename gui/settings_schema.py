from __future__ import annotations

import argparse
import math
import re
from collections import OrderedDict
from dataclasses import dataclass

from hybrid_infer_detail_preserving import parser as cli_parser

# GUI v0.5 deliberately exposes only the controls used by the public,
# user-facing workflows/tuning guidance in DOCUMENTATION.md.  The CLI still
# accepts every advanced parameter; hidden parameters retain parser defaults.
EXPOSED_DESTS = {
    # Restormer correction / runtime
    "restormer", "band_axis", "device", "amp", "processing_size", "passes",
    "first_pass_luma_strength", "first_pass_chroma_strength",
    "second_pass_strength", "luma_mode",
    # Local cleanup and tonal eligibility
    "flat_filter", "flat_luma_strength", "flat_chroma_strength", "flat_highpass", "flat_lowpass",
    # Residual profile
    "flat_profile", "flat_profile_mode", "flat_profile_luma_strength", "flat_profile_chroma_strength",
    "flat_profile_band_period", "flat_profile_pwm_polish", "flat_profile_pwm_polish_strength",
    "flat_profile_pwm_polish_passes",
    # Orthogonal residual-profile cleanup
    "orthogonal_profile", "orthogonal_profile_luma_strength", "orthogonal_profile_chroma_strength",
    # Documented local safety troubleshooting control
    "flat_local_edge_distance",
    # Broad/few-cycle cleanup
    "flat_surface_equalizer", "flat_surface_equalizer_mode", "flat_surface_equalizer_luma_strength",
    "flat_surface_equalizer_chroma_strength",
    # Tone restoration. max-gain and min-confidence stay CLI-only: the first is
    # a safety clamp nothing has come close to (measured gains 1.19x-1.33x
    # against a 1.6 cap), the second selects which period the stage smooths with
    # rather than whether it runs, and neither is something a user can reason
    # about from the panel.
    "tone_restore", "tone_restore_strength",
}

GROUP_ORDER = [
    "Restormer correction",
    "Flat-region cleanup",
    "Residual profile",
    "Broad residual cleanup",
    "Tone restoration",
]

SLIDERS = {
    "first_pass_luma_strength", "first_pass_chroma_strength",
    "second_pass_strength",
    "flat_luma_strength", "flat_chroma_strength",
    "flat_profile_luma_strength", "flat_profile_chroma_strength", "flat_profile_pwm_polish_strength",
    "orthogonal_profile_luma_strength", "orthogonal_profile_chroma_strength",
    "flat_surface_equalizer_luma_strength", "flat_surface_equalizer_chroma_strength",
    "tone_restore_strength",
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
    "flat_surface_equalizer_mode": ("dominant", "consensus"),
    "flat_profile_mode": ("smooth", "pwm"),
}

LABELS = {
    "restormer": "Enable Restormer correction",
    "device": "Device",
    "amp": "Use FP16 / AMP",
    "processing_size": "Processing size",
    "band_axis": "Band direction",
    "passes": "Passes",
    "first_pass_luma_strength": "First-pass luminance strength",
    "first_pass_chroma_strength": "First-pass chroma strength",
    "second_pass_strength": "Second-pass strength",
    "luma_mode": "Luminance mode",
    "flat_filter": "Enable flat-region cleanup",
    "flat_luma_strength": "Flat luminance strength",
    "flat_chroma_strength": "Flat chroma strength",
    "flat_highpass": "Shadow cutoff",
    "flat_lowpass": "Highlight cutoff",
    "flat_profile": "Enable residual profile",
    "flat_profile_mode": "Profile mode",
    "flat_profile_luma_strength": "Profile luminance strength",
    "flat_profile_chroma_strength": "Profile chroma strength",
    "flat_profile_band_period": "Profile band period (px)",
    "flat_profile_pwm_polish": "Enable final PWM polish",
    "flat_profile_pwm_polish_strength": "PWM polish strength",
    "flat_profile_pwm_polish_passes": "PWM polish max passes",
    "orthogonal_profile": "Enable orthogonal cleanup",
    "orthogonal_profile_luma_strength": "Orthogonal luminance strength",
    "orthogonal_profile_chroma_strength": "Orthogonal chroma strength",
    "flat_local_edge_distance": "Object-edge protection distance",
    "flat_surface_equalizer": "Enable broad residual cleanup",
    "flat_surface_equalizer_mode": "Broad mode",
    "flat_surface_equalizer_luma_strength": "Broad luminance strength",
    "flat_surface_equalizer_chroma_strength": "Broad chroma strength",
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
    if dest in {"restormer", "band_axis", "device", "amp", "processing_size", "passes", "first_pass_luma_strength", "first_pass_chroma_strength", "second_pass_strength", "luma_mode"}:
        return "Restormer correction"
    if dest in {"flat_filter", "flat_luma_strength", "flat_chroma_strength", "flat_highpass", "flat_lowpass", "flat_local_edge_distance"}:
        return "Flat-region cleanup"
    if dest.startswith("flat_profile") or dest.startswith("orthogonal_"):
        return "Residual profile"
    if dest in {"flat_surface_equalizer", "flat_surface_equalizer_mode", "flat_surface_equalizer_luma_strength", "flat_surface_equalizer_chroma_strength"}:
        return "Broad residual cleanup"
    if dest.startswith("tone_restore"):
        return "Tone restoration"
    return "Restormer correction"


# Explicit GUI ordering.  Do not rely on argparse declaration order or
# alphabetical dest sorting: this is the presentation order requested for the
# desktop application.
ITEM_ORDER = {
    # Tone restoration (rendered last; see GROUP_ORDER)
    "tone_restore": 900,
    "tone_restore_strength": 901,
    # Restormer correction
    "restormer": 0,
    "band_axis": 1,
    "device": 2,
    "amp": 3,
    "processing_size": 4,
    "passes": 5,
    "first_pass_luma_strength": 6,
    "first_pass_chroma_strength": 7,
    "second_pass_strength": 8,
    "luma_mode": 9,
    # Flat-region cleanup
    "flat_filter": 100,
    "flat_luma_strength": 101,
    "flat_chroma_strength": 102,
    "flat_highpass": 103,
    "flat_lowpass": 104,
    "flat_local_edge_distance": 105,
    # Residual profile + its optional perpendicular pass
    "flat_profile": 200,
    "flat_profile_mode": 201,
    "flat_profile_luma_strength": 202,
    "flat_profile_chroma_strength": 203,
    "flat_profile_band_period": 204,
    "flat_profile_pwm_polish": 205,
    "flat_profile_pwm_polish_strength": 206,
    "flat_profile_pwm_polish_passes": 207,
    "orthogonal_profile": 208,
    "orthogonal_profile_luma_strength": 209,
    "orthogonal_profile_chroma_strength": 210,
    # Broad residual cleanup
    "flat_surface_equalizer": 300,
    "flat_surface_equalizer_mode": 301,
    "flat_surface_equalizer_luma_strength": 302,
    "flat_surface_equalizer_chroma_strength": 303,
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
    if str(out.get("flat_profile_mode", "smooth")).lower() not in {"smooth", "pwm"}:
        out["flat_profile_mode"] = "smooth"
    else:
        out["flat_profile_mode"] = str(out.get("flat_profile_mode", "smooth")).lower()
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
    if "flat_profile_pwm_polish_passes" in out:
        try:
            v = int(out["flat_profile_pwm_polish_passes"])
        except Exception:
            v = 2
        out["flat_profile_pwm_polish_passes"] = max(1, min(6, v))
    for _key, _lo, _hi in (("tone_restore_strength", 0.0, 1.0),
                           ("tone_restore_max_gain", 1.01, 4.0),
                           ("tone_restore_min_confidence", 0.0, 1.0)):
        if _key in out:
            try:
                out[_key] = max(_lo, min(_hi, float(out[_key])))
            except (TypeError, ValueError):
                pass
    numeric_ranges = {
        "first_pass_luma_strength": (0.0, 2.0),
        "first_pass_chroma_strength": (0.0, 2.0),
        "second_pass_strength": (0.0, 2.0),
        "flat_luma_strength": (0.0, 2.0),
        "flat_chroma_strength": (0.0, 2.0),
        "flat_profile_luma_strength": (0.0, 4.0),
        "flat_profile_chroma_strength": (0.0, 4.0),
        "flat_profile_pwm_polish_strength": (0.0, 1.25),
        "orthogonal_profile_luma_strength": (-1.0, 4.0),
        "orthogonal_profile_chroma_strength": (-1.0, 4.0),
        "flat_surface_equalizer_luma_strength": (0.0, 2.0),
        "flat_surface_equalizer_chroma_strength": (0.0, 2.0),
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
    """Per-image GUI state contains only controls visible in the right panel.

    The desktop UI starts with Restormer disabled.  The CLI keeps its own
    parser default for backward compatibility; this override is intentionally
    GUI-only.
    """
    defaults = _all_parser_defaults()
    defaults["restormer"] = False
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



def validate_imported_settings(payload: object) -> OrderedDict[str, object]:
    """Validate a developer settings JSON payload and return a complete recipe.

    The importer accepts both current complete exports and older/partial exports:
    missing known keys are filled from the current defaults, while unknown keys,
    invalid JSON types, invalid choices, and non-finite numeric values are
    rejected with a human-readable ValueError.
    """
    if not isinstance(payload, dict):
        raise ValueError('The top-level JSON value must be an object containing processing settings.')
    if not payload:
        raise ValueError('The JSON object does not contain any processing settings.')

    actions = {
        action.dest: action
        for action in cli_parser()._actions
        if action.dest not in PLUMBING_DESTS
    }
    unknown = sorted(str(key) for key in payload.keys() if key not in actions)
    if unknown:
        preview=', '.join(unknown[:8])
        if len(unknown)>8:
            preview += f', ... (+{len(unknown)-8} more)'
        raise ValueError(f'Unknown processing setting(s): {preview}')

    # Fill missing values from the current GUI recipe plus authoritative hidden
    # parser defaults. This keeps older exported JSON files forward-compatible.
    merged = dict(_all_parser_defaults())
    merged.update(default_settings())

    gui_ranges = {
        'first_pass_luma_strength': (0.0, 2.0),
        'first_pass_chroma_strength': (0.0, 2.0),
        'second_pass_strength': (0.0, 2.0),
        'flat_luma_strength': (0.0, 2.0),
        'flat_chroma_strength': (0.0, 2.0),
        'flat_profile_luma_strength': (0.0, 4.0),
        'flat_profile_chroma_strength': (0.0, 4.0),
        'flat_profile_pwm_polish_strength': (0.0, 1.25),
        'orthogonal_profile_luma_strength': (-1.0, 4.0),
        'orthogonal_profile_chroma_strength': (-1.0, 4.0),
        'flat_surface_equalizer_luma_strength': (0.0, 2.0),
        'flat_surface_equalizer_chroma_strength': (0.0, 2.0),
    }

    for key, value in payload.items():
        action=actions[key]
        default=action.default
        is_bool=(
            isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction))
            or isinstance(default, bool)
        )
        if is_bool:
            if type(value) is not bool:
                raise ValueError(f'"{key}" must be true or false.')
            normalized=value
        elif action.type is int or (action.type is None and isinstance(default, int) and not isinstance(default, bool)):
            if type(value) is not int:
                raise ValueError(f'"{key}" must be an integer.')
            normalized=int(value)
        elif action.type is float or isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'"{key}" must be a number.')
            normalized=float(value)
            if not math.isfinite(normalized):
                raise ValueError(f'"{key}" must be a finite number.')
        else:
            if not isinstance(value, str):
                raise ValueError(f'"{key}" must be a string.')
            normalized=value

        if action.choices is not None and normalized not in action.choices:
            choices=', '.join(map(str, action.choices))
            raise ValueError(f'"{key}" must be one of: {choices}.')

        if key == 'device':
            text=str(normalized).strip().lower()
            if not re.fullmatch(r'(auto|cpu|mps|cuda(?::\d+)?)', text):
                raise ValueError('"device" must be auto, cpu, mps, cuda, or cuda:<index>.')
            normalized=text
        elif key == 'processing_size':
            if not 256 <= normalized <= 2048 or normalized % 8 != 0:
                raise ValueError('"processing_size" must be a multiple of 8 from 256 to 2048.')
        elif key == 'flat_local_edge_distance':
            if not 0 <= normalized <= 100:
                raise ValueError('"flat_local_edge_distance" must be from 0 to 100.')
        elif key == 'flat_profile_band_period':
            if normalized != 0.0 and not 1.0 <= normalized <= 7680.0:
                raise ValueError('"flat_profile_band_period" must be 0 (Auto) or from 1 to 7680 pixels.')
        elif key == 'tone_restore_strength':
            if not 0.0 <= normalized <= 1.0:
                raise ValueError('"tone_restore_strength" must be from 0 to 1.')
        elif key == 'tone_restore_max_gain':
            if not 1.01 <= normalized <= 4.0:
                raise ValueError('"tone_restore_max_gain" must be from 1.01 to 4.')
        elif key == 'tone_restore_min_confidence':
            if not 0.0 <= normalized <= 1.0:
                raise ValueError('"tone_restore_min_confidence" must be from 0 to 1.')
        elif key == 'flat_profile_pwm_polish_passes':
            if not 1 <= normalized <= 6:
                raise ValueError('"flat_profile_pwm_polish_passes" must be from 1 to 6.')
        elif key in gui_ranges:
            low, high=gui_ranges[key]
            if not low <= normalized <= high:
                raise ValueError(f'"{key}" must be from {low:g} to {high:g}.')
        elif key in {'flat_highpass','flat_lowpass'}:
            text=str(normalized).strip()
            raw=text[1:] if text.startswith('#') else text
            if not HEX_RE.fullmatch(raw):
                raise ValueError(f'"{key}" must be an RGB hex color such as #232323.')
            normalized='#'+raw.upper()

        merged[key]=normalized

    shadow=canonical_cutoff(merged.get('flat_highpass','#232323'),'#000000')
    highlight=canonical_cutoff(merged.get('flat_lowpass','#efefef'),'#FFFFFF')
    if cutoff_luma(shadow)>cutoff_luma(highlight):
        raise ValueError('"flat_highpass" (Shadow cutoff) must not be brighter than "flat_lowpass" (Highlight cutoff).')

    # Reuse the exact inference merge/policy path so imported settings have the
    # same normalization and hidden defaults as exported settings.
    complete=vars(namespace(merged))
    return OrderedDict((key, complete[key]) for key in _all_parser_defaults().keys())
