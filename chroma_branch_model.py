#!/usr/bin/env python3
"""Dedicated Cb/Cr Restormer branch.

The trunk is identical to the single-image Restormer, but the final head has
only two channels. The branch therefore predicts centered Cb and Cr directly
and has no route to control final luminance in the two-component inference
pipeline.
"""
from __future__ import annotations

from restormer_model import Restormer


def build_chroma_branch() -> Restormer:
    return Restormer(inp_channels=3, out_channels=2)
