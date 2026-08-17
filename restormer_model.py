#!/usr/bin/env python3
"""Reduced Restormer used by the BurstDeflicker baselines.

The architecture follows the official BurstDeflicker Restormer configuration:
  dim=32, blocks=(2,3,4,5), refinement_blocks=2, heads=(1,2,4,8).

The upstream BurstDeflicker implementation is Apache-2.0 licensed:
https://github.com/qulishen/BurstDeflicker
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=height, w=width)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, Sequence[int]]) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        shape = torch.Size(normalized_shape)
        if len(shape) != 1:
            raise ValueError("LayerNorm expects a one-dimensional normalized shape")
        self.weight = nn.Parameter(torch.ones(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, Sequence[int]]) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        shape = torch.Size(normalized_shape)
        if len(shape) != 1:
            raise ValueError("LayerNorm expects a one-dimensional normalized shape")
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str) -> None:
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body: nn.Module = BiasFreeLayerNorm(dim)
        elif layer_norm_type == "WithBias":
            self.body = WithBiasLayerNorm(dim)
        else:
            raise ValueError(f"Unsupported LayerNorm type: {layer_norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), height, width)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: float, bias: bool) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        out = attention @ v
        out = rearrange(
            out,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=height,
            w=width,
        )
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        expansion: float,
        bias: bool,
        layer_norm_type: str,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim, layer_norm_type)
        self.ffn = FeedForward(dim, expansion, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, bias: bool) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Restormer(nn.Module):
    """Reduced Restormer. Set ``inp_channels=3`` for the single-image model."""

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 32,
        num_blocks: Sequence[int] = (2, 3, 4, 5),
        num_refinement_blocks: int = 2,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
        dual_pixel_task: bool = False,
    ) -> None:
        super().__init__()
        if len(num_blocks) != 4 or len(heads) != 4:
            raise ValueError("num_blocks and heads must each contain four values")

        def block(features: int, num_heads: int) -> TransformerBlock:
            return TransformerBlock(
                features,
                num_heads,
                ffn_expansion_factor,
                bias,
                layer_norm_type,
            )

        self.inp_channels = inp_channels
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias)
        self.encoder_level1 = nn.Sequential(*[block(dim, heads[0]) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[block(dim * 2, heads[1]) for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.Sequential(*[block(dim * 4, heads[2]) for _ in range(num_blocks[2])])
        self.down3_4 = Downsample(dim * 4)
        self.latent = nn.Sequential(*[block(dim * 8, heads[3]) for _ in range(num_blocks[3])])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[block(dim * 4, heads[2]) for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[block(dim * 2, heads[1]) for _ in range(num_blocks[1])])

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(*[block(dim * 2, heads[0]) for _ in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[block(dim * 2, heads[0]) for _ in range(num_refinement_blocks)])

        self.dual_pixel_task = dual_pixel_task
        if dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
        return self.output(out_dec_level1)


def build_single_image_restormer() -> Restormer:
    return Restormer(inp_channels=3)


def build_burst_restormer() -> Restormer:
    return Restormer(inp_channels=9)


def _torch_load(path: Path, *, weights_only: bool) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint_object(path: Path) -> Any:
    """Load a checkpoint, preferring PyTorch's restricted tensor-only loader."""
    try:
        return _torch_load(path, weights_only=True)
    except Exception as safe_error:
        try:
            return _torch_load(path, weights_only=False)
        except Exception as legacy_error:
            raise RuntimeError(f"Could not load checkpoint {path}. Safe load failed: {safe_error}. Legacy load failed: {legacy_error}") from legacy_error


def clean_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint).__name__}")

    candidate: Any = checkpoint
    for key in ("params_ema", "params", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            candidate = value
            break

    state: dict[str, torch.Tensor] = {}
    for raw_key, value in candidate.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            continue
        key = raw_key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "net_g.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        state[key] = value

    if not state:
        raise ValueError("The checkpoint contains no tensor state dictionary")
    return state


def load_state_dict_file(path: Path) -> dict[str, torch.Tensor]:
    return clean_state_dict(load_checkpoint_object(path))


def find_patch_embed_key(state: Mapping[str, torch.Tensor]) -> str:
    exact = "patch_embed.proj.weight"
    if exact in state:
        return exact
    matches = [key for key in state if key.endswith(exact)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"Checkpoint has no {exact!r} tensor")
    raise KeyError(f"Checkpoint has multiple possible patch-embedding tensors: {matches}")


def checkpoint_input_channels(state: Mapping[str, torch.Tensor]) -> int:
    key = find_patch_embed_key(state)
    weight = state[key]
    if weight.ndim != 4:
        raise ValueError(f"{key} should be a 4-D convolution weight, got {tuple(weight.shape)}")
    return int(weight.shape[1])


def strict_load(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as error:
        raise RuntimeError(f"Checkpoint is incompatible with this Restormer architecture: {error}") from error


def choose_device(requested: str) -> torch.device:
    requested = str(requested).strip().lower()
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        index = 0 if device.index is None else int(device.index)
        count = int(torch.cuda.device_count())
        if index < 0 or index >= count:
            raise RuntimeError(f"CUDA device {index} was requested, but only {count} CUDA device(s) are available")
        device = torch.device(f"cuda:{index}")
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return device
