from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceOption:
    label: str
    value: str


def _clean_name(value: str | None) -> str:
    text = " ".join(str(value or "").strip().split())
    return text


def cpu_model_name() -> str:
    """Best-effort CPU marketing/model name without adding dependencies."""
    if sys.platform.startswith("win"):
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                name = _clean_name(value)
                if name:
                    return name
        except Exception:
            pass

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.lower().startswith("model name") and ":" in line:
                        name = _clean_name(line.split(":", 1)[1])
                        if name:
                            return name
        except OSError:
            pass

    for value in (
        platform.processor(),
        os.environ.get("PROCESSOR_IDENTIFIER"),
        platform.machine(),
    ):
        name = _clean_name(value)
        if name:
            return name
    return "CPU"


def _cuda_names() -> list[str]:
    if not torch.cuda.is_available():
        return []
    names: list[str] = []
    try:
        count = int(torch.cuda.device_count())
    except Exception:
        count = 0
    for index in range(max(0, count)):
        try:
            name = _clean_name(torch.cuda.get_device_name(index)) or f"CUDA GPU {index}"
        except Exception:
            name = f"CUDA GPU {index}"
        names.append(name)
    return names


def device_options() -> list[DeviceOption]:
    """Return GUI device labels mapped to exact torch device strings."""
    cpu_name = cpu_model_name()
    cuda_names = _cuda_names()
    mps_available = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )

    if cuda_names:
        auto_label = f"auto ({cuda_names[0]})"
    elif mps_available:
        auto_label = "auto (mps)"
    else:
        auto_label = f"auto ({cpu_name})"

    result = [DeviceOption(auto_label, "auto")]
    if len(cuda_names) == 1:
        result.append(DeviceOption(f"cuda ({cuda_names[0]})", "cuda:0"))
    else:
        duplicate_names = len(set(cuda_names)) != len(cuda_names)
        for index, name in enumerate(cuda_names):
            # Match the compact label requested when model names are already
            # unique. If identical GPUs are installed, expose the CUDA index so
            # the two otherwise-identical entries remain distinguishable.
            label = f"cuda:{index} ({name})" if duplicate_names else f"cuda ({name})"
            result.append(DeviceOption(label, f"cuda:{index}"))
    result.append(DeviceOption(f"cpu ({cpu_name})", "cpu"))
    if mps_available:
        result.append(DeviceOption("mps", "mps"))
    return result
