from __future__ import annotations

import json
import sys
from pathlib import Path


def app_root() -> Path:
    if "__compiled__" in globals() or getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_path(*parts: str) -> Path:
    return app_root().joinpath(*parts)


def state_path() -> Path:
    return app_root() / "FlickerSuppressor.settings.json"


def load_state() -> dict:
    try:
        p = state_path()
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def save_state(state: dict) -> None:
    try:
        state_path().write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
