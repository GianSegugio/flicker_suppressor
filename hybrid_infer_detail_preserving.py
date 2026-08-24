#!/usr/bin/env python3
"""Two-component restoration with directional Y correction and optional flat-area cleanup.

Features in v1.9:
- v1.5 neural/local-flat behavior is preserved.
- Optional pass-2 exposure/DC lock prevents recursive luminance accumulation.
- Local flat-region cleanup remains edge-aware and texture-preserving.
- v1.7 local/profile/surface-equalizer behavior is preserved.
- Auto/manual band-axis normalization rotates visible vertical bands into the
  row-oriented domain used by the trained models and all directional filters.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from band_axis import decide_band_axis, orient_for_processing, restore_display_orientation
from chroma_branch_model import build_chroma_branch
from color_space import rgb_to_y_cbcr, y_cbcr_to_rgb_preserve_y
from correction_field import apply_correction_field, make_correction_field, remove_global_dc
from flat_region_filter import apply_flat_region_filter
from restormer_model import build_single_image_restormer, choose_device, load_state_dict_file, strict_load

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class PassInfo:
    field: torch.Tensor
    chroma_delta: torch.Tensor
    raw_rms: float
    constrained_rms: float
    removed_rms: float
    exposure_removed_stops: float
    gamut_compressed: float


def to_tensor(im: Image.Image) -> torch.Tensor:
    a = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()


def save_rgb(x: torch.Tensor, path: Path) -> None:
    a = x.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a, mode="RGB").save(path, format="PNG")


def save_gray(x: torch.Tensor, path: Path) -> None:
    a = x.clamp(0, 1).squeeze().mul(255).round().byte().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a, mode="L").save(path, format="PNG")


def save_field_map(field: torch.Tensor, path: Path, domain: str) -> None:
    x = field.detach().float()
    if domain == "log":
        x = 0.5 + 0.5 * x / np.log(2.0)
    else:
        x = 0.5 + 0.5 * x / 0.20
    save_gray(x, path)


def resize(x: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)


def autocast(device: torch.device, enabled: bool):
    return torch.autocast("cuda", dtype=torch.float16) if enabled and device.type == "cuda" else contextlib.nullcontext()


from tone_restore import match_tone_log_torch


def discover(p: Path):
    if p.is_file():
        return [p]
    return sorted(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in EXTS)


def out_for(src: Path, root: Path, out: Path) -> Path:
    if root.is_file():
        return out.with_suffix(".png") if out.suffix else out / (src.stem + ".png")
    return (out / src.relative_to(root)).with_suffix(".png")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hybrid Restormer with detail-preserving Y and optional residual-band cleanup.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--luma-model", type=Path, default=None, help="Original/standard 3-channel Restormer final.pth (required unless --no-restormer)")
    p.add_argument("--chroma-model", type=Path, default=None, help="Fine-tuned 2-channel CbCr branch (required unless --no-restormer)")
    p.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, mps, cuda, or an indexed CUDA device such as cuda:1")
    p.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="Use CUDA FP16 autocast when running on CUDA (default: on; use --no-amp to disable)",
    )
    p.add_argument("--processing-size", type=int, default=512)
    p.add_argument("--band-axis", choices=("auto", "horizontal", "vertical", "both"), default="auto",
                   help="Visible primary band direction. 'both' is a legacy shortcut for horizontal + --orthogonal-profile")
    p.add_argument("--band-axis-auto-aspect-ratio", type=float, default=1.10,
                   help="Portrait/landscape H/W ratio used by --band-axis auto; near-square images use a coarse striping score")
    p.add_argument("--band-axis-analysis-size", type=int, default=384,
                   help="Maximum side used only for the near-square auto-axis diagnostic")
    p.add_argument("--orthogonal-profile", action="store_true",
                   help="Enable a residual-profile cleanup in the axis perpendicular to the primary band direction")
    p.add_argument("--orthogonal-profile-luma-strength", type=float, default=-1.0,
                   help="Orthogonal-profile Y strength; negative = reuse --flat-profile-luma-strength")
    p.add_argument("--orthogonal-profile-chroma-strength", type=float, default=-1.0,
                   help="Orthogonal-profile CbCr strength; negative = reuse --flat-profile-chroma-strength")
    p.add_argument("--orthogonal-profile-band-period", type=float, default=0.0,
                   help="Orthogonal-profile period override in perpendicular-axis pixels; 0 = auto")

    p.add_argument(
        "--restormer", action=argparse.BooleanOptionalAction, default=True,
        help="Enable the neural Restormer Y/CbCr correction stage (use --no-restormer to run only deterministic cleanup stages)",
    )
    p.add_argument("--passes", type=int, choices=(1, 2), default=1,
                   help="Use 2 only for unusually severe residual bands")
    p.add_argument("--first-pass-luma-strength", type=float, default=1.0,
                   help="Pass-1-only multiplier for the Restormer luminance correction")
    p.add_argument("--first-pass-chroma-strength", type=float, default=1.0,
                   help="Pass-1-only multiplier for the Restormer CbCr correction")
    p.add_argument("--second-pass-strength", type=float, default=1.0,
                   help="Scale both Y and chroma corrections on pass 2")
    p.add_argument("--exposure-lock", choices=("off", "pass2", "all"), default="pass2",
                   help="Remove global DC from the Y correction; pass2 prevents recursive brightness drift")

    p.add_argument("--luma-mode", choices=("raw", "directional", "directional-additive", "row"), default="directional")
    p.add_argument("--horizontal-sigma", type=float, default=16.0)
    p.add_argument("--vertical-sigma", type=float, default=0.0)
    p.add_argument("--luma-eps", type=float, default=0.02)
    p.add_argument("--clip-stops", type=float, default=2.0)
    p.add_argument("--no-row-anchor", action="store_true")
    p.add_argument("--luma-strength", type=float, default=1.0)
    p.add_argument("--chroma-strength", type=float, default=1.0)
    p.add_argument("--tone-restore", action=argparse.BooleanOptionalAction, default=True,
                   help="Restore global contrast/gamma lost during processing")
    p.add_argument("--tone-restore-strength", type=float, default=1.0,
                   help="Restore global contrast/gamma lost during processing by matching the "
                        "output tone curve to the source, fitted on band-axis-smoothed envelopes "
                        "so the band residual is untouched; 0 disables")
    p.add_argument("--tone-restore-max-gain", type=float, default=1.6,
                   help="Largest luminance gain the tone restoration may apply at any level")
    p.add_argument("--tone-restore-min-confidence", type=float, default=0.35,
                   help="Skip tone restoration when band confidence is below this; protects "
                        "images with non-stationary periods where band-axis smoothing is invalid")

    p.add_argument("--flat-filter", action="store_true",
                   help="Enable optional flat-region residual band post-filter")
    p.add_argument("--flat-band-period", type=float, default=0.0,
                   help="Band period in full-resolution pixels; 0 = estimate from model corrections")
    p.add_argument("--flat-period-sigma-ratio", type=float, default=0.25,
                   help="Vertical smoothing sigma as a fraction of estimated band period")
    p.add_argument("--flat-horizontal-sigma", type=float, default=16.0,
                   help="Horizontal coherence support in full-resolution pixels")
    p.add_argument("--flat-full", type=float, default=0.007,
                   help="Detail metric below this gets full flat-region weight")
    p.add_argument("--flat-none", type=float, default=0.025,
                   help="Detail metric above this gets zero flat-region weight")
    p.add_argument("--flat-luma-strength", type=float, default=0.70,
                   help="Strength of the local flat-region luminance correction")
    p.add_argument("--flat-chroma-strength", type=float, default=0.85,
                   help="Strength of the local flat-region Cb/Cr correction")
    p.add_argument("--flat-cleanup-strength", type=float, default=1.0,
                   help="Legacy CLI multiplier for both flat local strengths; the GUI keeps this at 1.0")
    p.add_argument("--flat-highpass", default="#232323",
                   help="Dark-side Lab-lightness cutoff; darker tones are excluded. Use quoted #RRGGBB or 'off'.")
    p.add_argument("--flat-lowpass", default="#efefef",
                   help="Bright-side Lab-lightness cutoff; brighter tones are excluded. Use quoted #RRGGBB or 'off'.")
    p.add_argument("--flat-shadow-ramp", type=float, default=12.0,
                   help="Smooth shadow transition width around --flat-highpass in CIE Lab L* units")
    p.add_argument("--flat-highlight-ramp", type=float, default=8.0,
                   help="Smooth highlight transition width around --flat-lowpass in CIE Lab L* units")
    p.add_argument("--flat-luma-spatial-feather", type=float, default=1.25,
                   help="Small spatial feather sigma for the tone gate; 0 disables")
    p.add_argument("--flat-base-lstar-sigma-ratio", type=float, default=0.40,
                   help="Vertical smoothing scale for band-resistant base lightness as fraction of band period")
    p.add_argument("--flat-edge-low", type=float, default=0.018,
                   help="Scene-edge barrier starts at this smoothed Y/C gradient")
    p.add_argument("--flat-edge-high", type=float, default=0.055,
                   help="Scene-edge barrier is fully closed at this smoothed Y/C gradient")
    p.add_argument("--flat-edge-guard", type=int, default=2,
                   help="Pixels of support-only guard around strong scene edges")
    p.add_argument("--flat-broad-structure-sigma", type=float, default=8.0,
                   help="Broad scale used to detect soft/defocused scene boundaries for local flat cleanup")
    p.add_argument("--flat-broad-structure-low", type=float, default=0.025,
                   help="Scale-normalized broad-gradient value where soft-structure protection begins")
    p.add_argument("--flat-broad-structure-high", type=float, default=0.080,
                   help="Scale-normalized broad-gradient value where soft-structure protection is full")
    p.add_argument("--flat-broad-structure-guard", type=int, default=8,
                   help="Pixels of extra local-flat guard around detected soft/defocused structure")
    p.add_argument("--flat-broad-structure-feather", type=float, default=6.0,
                   help="Feather sigma for the soft/defocused structure guard")
    p.add_argument("--flat-coarse-preblur", type=float, default=1.0,
                   help="Preblur used only to segment textured close-colored surfaces")
    p.add_argument("--flat-coarse-blend", type=float, default=0.25,
                   help="How much coarse surface segmentation contributes to the visible local blend")
    p.add_argument("--flat-blend-feather", type=float, default=0.75,
                   help="Spatial feather sigma for the final correction blend mask")
    p.add_argument("--flat-local-extent-sigma", type=float, default=32.0,
                   help="Large-neighborhood scale used to reject tiny isolated local-flat islands such as smooth skin patches")
    p.add_argument("--flat-local-extent-low", type=float, default=0.18,
                   help="Large-surface occupancy where local flattening begins to fade in")
    p.add_argument("--flat-local-extent-high", type=float, default=0.50,
                   help="Large-surface occupancy where local flattening is fully enabled")
    p.add_argument("--flat-local-color-sigma", type=float, default=44.0,
                   help="Broad neighborhood scale for same-surface color consistency in the local flattener")
    p.add_argument("--flat-local-color-preblur", type=float, default=2.0,
                   help="Preblur before local same-surface color comparison")
    p.add_argument("--flat-local-color-luma-tolerance", type=float, default=7.0,
                   help="Allowed broad-surface Lab L* deviation for local flattening")
    p.add_argument("--flat-local-color-chroma-tolerance", type=float, default=0.030,
                   help="Allowed broad-surface CbCr distance for local flattening")
    p.add_argument("--flat-local-fill-sigma", type=float, default=36.0,
                   help="Neighborhood scale for same-surface fill ratio")
    p.add_argument("--flat-local-fill-low", type=float, default=0.38,
                   help="Same-surface fill fraction where local flattening begins to fade in")
    p.add_argument("--flat-local-fill-high", type=float, default=0.68,
                   help="Same-surface fill fraction where local flattening is fully enabled")
    p.add_argument("--flat-local-edge-distance", type=int, default=6,
                   help="Local-filter-only guard distance from strong scene/object edges")
    p.add_argument("--flat-local-edge-feather", type=float, default=3.0,
                   help="Soft feather applied to the local edge-distance safety gate")
    p.add_argument("--flat-local-correction-horizontal-sigma", type=float, default=128.0,
                   help="Horizontal regularization sigma for the local correction field; reduces object-silhouette trails")
    p.add_argument("--flat-local-application-horizontal-sigma", type=float, default=48.0,
                   help="Horizontal regularization sigma for the final blended local correction; reduces edge halos around protected objects")
    p.add_argument("--flat-profile", action="store_true",
                   help="Enable optional residual 1-D row-profile suppression for faint globally coherent bands")
    p.add_argument("--flat-profile-mode", choices=("smooth", "pwm"), default="smooth",
                   help="Residual-profile waveform model: smooth periodic, or sharp two-state PWM/step")
    p.add_argument("--flat-profile-luma-strength", type=float, default=0.35,
                   help="Residual row-profile suppression strength for log-Y when --flat-profile is enabled")
    p.add_argument("--flat-profile-chroma-strength", type=float, default=0.35,
                   help="Residual row-profile suppression strength for CbCr when --flat-profile is enabled")
    p.add_argument("--flat-profile-narrow-ratio", type=float, default=0.035,
                   help="Noise-suppression scale of residual profile as fraction of band period")
    p.add_argument("--flat-profile-base-ratio", type=float, default=0.40,
                   help="Baseline scale of residual profile as fraction of band period")
    p.add_argument("--flat-profile-pwm-transition-ratio", type=float, default=0.010,
                   help="PWM/step transition feather as fraction of the detected period")
    p.add_argument("--flat-profile-pwm-min-duty", type=float, default=0.08,
                   help="Minimum accepted PWM plateau duty fraction")
    p.add_argument("--flat-profile-pwm-max-duty", type=float, default=0.92,
                   help="Maximum accepted PWM plateau duty fraction")
    p.add_argument("--flat-profile-pwm-min-transition-score", type=float, default=2.0,
                   help="Minimum normalized repeated-transition score before PWM mode is accepted")
    p.add_argument("--flat-profile-band-period", type=float, default=0.0,
                   help="Override only the residual-profile period in full-resolution pixels; 0 = multiscale auto")
    p.add_argument("--flat-profile-period-mode", choices=("multiscale", "legacy"), default="multiscale",
                   help="Period estimator for --flat-profile; multiscale is texture tolerant and harmonic aware")
    p.add_argument("--flat-profile-period-min", type=float, default=12.0,
                   help="Smallest period considered by multiscale profile detection")
    p.add_argument("--flat-profile-period-max-fraction", type=float, default=0.60,
                   help="Largest profile period as a fraction of image height")
    p.add_argument("--flat-profile-period-analysis", type=int, default=512,
                   help="Internal row-spectrum analysis height")
    p.add_argument("--flat-profile-huber-k", type=float, default=2.5,
                   help="Robust row-consensus outlier scale; lower rejects objects/texture more aggressively")
    p.add_argument("--flat-profile-min-coverage", type=float, default=0.08,
                   help="Minimum trustworthy horizontal support fraction for a row profile")
    p.add_argument("--flat-profile-adaptive", action=argparse.BooleanOptionalAction, default=True,
                   help="Fit a slowly varying local amplitude for the globally estimated residual profile")
    p.add_argument("--flat-profile-adaptive-x-ratio", type=float, default=0.35,
                   help="Horizontal adaptive-amplitude smoothing scale as a fraction of band period")
    p.add_argument("--flat-profile-adaptive-y-ratio", type=float, default=1.50,
                   help="Vertical adaptive-amplitude fitting scale as a fraction of band period")
    p.add_argument("--flat-profile-adaptive-corr-low", type=float, default=0.15,
                   help="Local band-limited waveform-correlation value where adaptive profile evidence begins")
    p.add_argument("--flat-profile-adaptive-corr-high", type=float, default=0.45,
                   help="Local band-limited waveform-correlation value where adaptive profile evidence is fully trusted")
    p.add_argument("--flat-profile-adaptive-max-gain", type=float, default=1.00,
                   help="Maximum local multiplier of the globally estimated residual-profile waveform; 1.0 means attenuation-only")
    p.add_argument("--flat-profile-no-harm", action=argparse.BooleanOptionalAction, default=True,
                   help="Locally suppress residual-profile correction where it would increase band-limited residual energy")
    p.add_argument("--flat-profile-pwm-polish", action=argparse.BooleanOptionalAction, default=False,
                   help="Optional final PWM-only polish using only already-validated period/phase families")
    p.add_argument("--flat-profile-pwm-polish-strength", type=float, default=1.0,
                   help="Final PWM polish authority; 1.0 subtracts the measured remaining PWM component")
    p.add_argument("--flat-profile-pwm-polish-passes", type=int, default=2,
                   help="Maximum accepted PWM polish passes (1-3); each pass must reduce exact-mode energy")

    p.add_argument("--flat-surface-equalizer", action="store_true",
                   help="Enable broad/few-cycle residual cleanup")
    p.add_argument("--flat-surface-equalizer-mode", choices=("dominant", "consensus"), default="consensus",
                   help="Broad cleanup mode: one dominant surface, or multi-surface Y/CbCr consensus")
    p.add_argument("--flat-surface-equalizer-luma-strength", type=float, default=1.0,
                   help="Broad log-luminance strength; consensus mode treats this as a maximum no-harm authority")
    p.add_argument("--flat-surface-equalizer-chroma-strength", type=float, default=1.0,
                   help="Broad CbCr strength; consensus mode treats this as a maximum vector no-harm authority")
    p.add_argument("--flat-surface-equalizer-degree", type=int, default=2,
                   help="Polynomial degree of the legitimate large-surface illumination/color baseline (default quadratic)")
    p.add_argument("--flat-surface-equalizer-preblur", type=float, default=4.0,
                   help="Preblur used to recognize a rough textured surface as one underlying surface")
    p.add_argument("--flat-surface-equalizer-analysis", type=int, default=256,
                   help="Maximum side of low-resolution dominant-surface connected-component analysis")
    p.add_argument("--flat-surface-equalizer-threshold", type=float, default=0.30,
                   help="Low-resolution candidate threshold for the dominant surface")
    p.add_argument("--flat-surface-equalizer-close-radius", type=int, default=1,
                   help="Small low-resolution morphological closing radius for holes in textured surfaces")
    p.add_argument("--flat-surface-equalizer-min-area", type=float, default=0.08,
                   help="Minimum image-area fraction required before a surface can be equalized")
    p.add_argument("--flat-surface-equalizer-chroma-tolerance", type=float, default=0.080,
                   help="CbCr distance tolerance used to keep the dominant large surface color-consistent")
    p.add_argument("--flat-surface-equalizer-luma-tolerance", type=float, default=0.24,
                   help="Coarse Y tolerance used to separate the dominant surface from different objects")
    p.add_argument("--flat-surface-equalizer-row-edge-barrier", type=float, default=0.030,
                   help="Row-spanning scene-edge density that splits otherwise reconnecting large surfaces")
    p.add_argument("--flat-surface-equalizer-row-edge-guard", type=int, default=3,
                   help="Vertical pixels guarded around a row-spanning surface boundary")
    p.add_argument("--flat-surface-equalizer-feather", type=float, default=5.0,
                   help="Full-resolution feather sigma of the selected surface boundary")
    p.add_argument("--flat-surface-equalizer-row-sigma", type=float, default=2.0,
                   help="Small 1-D smoothing of robust per-row surface measurements before baseline fitting")
    p.add_argument("--flat-surface-equalizer-huber-k", type=float, default=2.5,
                   help="Robust cross-column outlier scale for the large-surface row estimate")
    p.add_argument("--flat-surface-equalizer-min-coverage", type=float, default=0.04,
                   help="Minimum horizontal coverage of the selected surface required on a row")
    p.add_argument("--flat-broad-consensus-regions", type=int, default=6,
                   help="Number of large processing-X regions used by multi-surface broad consensus")
    p.add_argument("--flat-broad-consensus-min-regions", type=int, default=2,
                   help="Minimum mutually agreeing regions required by multi-surface broad consensus")
    p.add_argument("--flat-broad-consensus-corr-low", type=float, default=0.55,
                   help="Neural/residual anti-correlation where broad-consensus confidence begins")
    p.add_argument("--flat-broad-consensus-corr-high", type=float, default=0.80,
                   help="Neural/residual anti-correlation where broad-consensus confidence is full")
    p.add_argument("--flat-broad-consensus-smooth-fraction", type=float, default=0.015,
                   help="Broad-consensus row smoothing sigma as a fraction of processing height")
    p.add_argument("--flat-broad-consensus-baseline-fraction", type=float, default=0.20,
                   help="Legacy/fallback very-slow baseline sigma; neural-guided consensus uses affine region baselines")

    p.add_argument("--flat-allow-mean-shift", action="store_true",
                   help="Allow the local flat filter to change global Y/CbCr means")

    p.add_argument("--debug-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p


def run_pass(
    current: torch.Tensor,
    *,
    luma,
    chroma,
    device: torch.device,
    amp: bool,
    args,
    pass_index: int,
) -> tuple[torch.Tensor, PassInfo, str]:
    h, w = current.shape[-2:]
    small = resize(current, (args.processing_size, args.processing_size))
    with torch.inference_mode(), autocast(device, amp):
        luma_rgb = luma(small)
        chroma_pred = chroma(small)

    small_y, small_c = rgb_to_y_cbcr(small.float())
    pred_y, _ = rgb_to_y_cbcr(luma_rgb.float())
    domain, field, stats = make_correction_field(
        small_y,
        pred_y,
        mode=args.luma_mode,
        horizontal_sigma=args.horizontal_sigma,
        vertical_sigma=args.vertical_sigma,
        eps=args.luma_eps,
        clip_stops=args.clip_stops,
        row_anchor=not args.no_row_anchor,
    )

    lock = args.exposure_lock == "all" or (args.exposure_lock == "pass2" and pass_index == 2)
    removed_stops = 0.0
    if lock:
        field, removed_stops = remove_global_dc(field, small_y, domain=domain)

    if pass_index == 1:
        luma_scale = float(args.luma_strength) * float(args.first_pass_luma_strength)
        chroma_scale = float(args.chroma_strength) * float(args.first_pass_chroma_strength)
    else:
        # Keep the historical pass-2 control independent of the new pass-1
        # tuning sliders. Hidden legacy luma/chroma strengths remain global
        # compatibility multipliers for CLI/settings files.
        luma_scale = float(args.luma_strength) * float(args.second_pass_strength)
        chroma_scale = float(args.chroma_strength) * float(args.second_pass_strength)
    current_y, current_c = rgb_to_y_cbcr(current.float())
    final_y = apply_correction_field(
        current_y,
        field,
        domain=domain,
        eps=args.luma_eps,
        strength=luma_scale,
    )
    dc = (chroma_pred.float() - small_c) * chroma_scale
    dc_full = resize(dc, (h, w))
    final, gamut_alpha = y_cbcr_to_rgb_preserve_y(final_y, current_c + dc_full)
    compressed = float((gamut_alpha < 0.999).float().mean())
    info = PassInfo(
        field=field.detach(),
        chroma_delta=dc.detach(),
        raw_rms=stats.raw_rms,
        constrained_rms=float(field.float().square().mean().sqrt()),
        removed_rms=stats.removed_rms,
        exposure_removed_stops=removed_stops,
        gamut_compressed=compressed,
    )
    return final, info, domain



def run_orthogonal_profile_cleanup(
    current: torch.Tensor,
    *,
    args,
) -> tuple[torch.Tensor, object, dict]:
    """Apply only the robust residual-profile stage in the orthogonal axis.

    This intentionally does not run another neural pass and sets the local flat
    filter strengths to zero.  It therefore preserves the already-good primary
    restoration while attacking striping coherent along the other image axis.
    """
    rotated = torch.rot90(current, 1, dims=(-2, -1))
    y, c = rgb_to_y_cbcr(rotated.float())
    debug = {}
    y2, c2, _mask, stats = apply_flat_region_filter(
        y, c,
        luma_field=None,
        chroma_delta_hint=None,
        band_period_px=0.0,
        period_sigma_ratio=args.flat_period_sigma_ratio,
        coherence_sigma_x=args.flat_horizontal_sigma,
        flat_full=args.flat_full,
        flat_none=args.flat_none,
        luma_strength=0.0,
        chroma_strength=0.0,
        luma_highpass=args.flat_highpass,
        luma_lowpass=args.flat_lowpass,
        shadow_ramp_lstar=args.flat_shadow_ramp,
        highlight_ramp_lstar=args.flat_highlight_ramp,
        luma_spatial_feather=args.flat_luma_spatial_feather,
        base_lstar_sigma_ratio=args.flat_base_lstar_sigma_ratio,
        edge_low=args.flat_edge_low,
        edge_high=args.flat_edge_high,
        edge_guard_px=args.flat_edge_guard,
        broad_structure_sigma=args.flat_broad_structure_sigma,
        broad_structure_low=args.flat_broad_structure_low,
        broad_structure_high=args.flat_broad_structure_high,
        broad_structure_guard_px=args.flat_broad_structure_guard,
        broad_structure_feather=args.flat_broad_structure_feather,
        coarse_preblur_sigma=args.flat_coarse_preblur,
        coarse_blend_weight=0.0,
        blend_feather=args.flat_blend_feather,
        local_extent_sigma=args.flat_local_extent_sigma,
        local_extent_low=args.flat_local_extent_low,
        local_extent_high=args.flat_local_extent_high,
        local_color_sigma=args.flat_local_color_sigma,
        local_color_preblur=args.flat_local_color_preblur,
        local_color_luma_tolerance=args.flat_local_color_luma_tolerance,
        local_color_chroma_tolerance=args.flat_local_color_chroma_tolerance,
        local_fill_sigma=args.flat_local_fill_sigma,
        local_fill_low=args.flat_local_fill_low,
        local_fill_high=args.flat_local_fill_high,
        local_edge_distance=args.flat_local_edge_distance,
        local_edge_feather=args.flat_local_edge_feather,
        local_correction_horizontal_sigma=args.flat_local_correction_horizontal_sigma,
        local_application_horizontal_sigma=args.flat_local_application_horizontal_sigma,
        residual_profile_luma_strength=(args.flat_profile_luma_strength if args.orthogonal_profile_luma_strength < 0 else args.orthogonal_profile_luma_strength),
        residual_profile_chroma_strength=(args.flat_profile_chroma_strength if args.orthogonal_profile_chroma_strength < 0 else args.orthogonal_profile_chroma_strength),
        residual_profile_mode=args.flat_profile_mode,
        residual_profile_narrow_ratio=args.flat_profile_narrow_ratio,
        residual_profile_pwm_transition_ratio=args.flat_profile_pwm_transition_ratio,
        residual_profile_pwm_min_duty=args.flat_profile_pwm_min_duty,
        residual_profile_pwm_max_duty=args.flat_profile_pwm_max_duty,
        residual_profile_pwm_min_transition_score=args.flat_profile_pwm_min_transition_score,
        residual_profile_base_ratio=args.flat_profile_base_ratio,
        residual_profile_band_period_px=args.orthogonal_profile_band_period,
        period_mode=args.flat_profile_period_mode,
        period_min_px=args.flat_profile_period_min,
        period_max_fraction=args.flat_profile_period_max_fraction,
        period_analysis_h=args.flat_profile_period_analysis,
        residual_profile_huber_k=args.flat_profile_huber_k,
        residual_profile_min_coverage=args.flat_profile_min_coverage,
        residual_profile_adaptive=args.flat_profile_adaptive,
        residual_profile_adaptive_x_ratio=args.flat_profile_adaptive_x_ratio,
        residual_profile_adaptive_y_ratio=args.flat_profile_adaptive_y_ratio,
        residual_profile_adaptive_corr_low=args.flat_profile_adaptive_corr_low,
        residual_profile_adaptive_corr_high=args.flat_profile_adaptive_corr_high,
        residual_profile_adaptive_max_gain=args.flat_profile_adaptive_max_gain,
        residual_profile_no_harm=args.flat_profile_no_harm,
        residual_profile_pwm_polish=args.flat_profile_pwm_polish,
        residual_profile_pwm_polish_strength=args.flat_profile_pwm_polish_strength,
        residual_profile_pwm_polish_passes=args.flat_profile_pwm_polish_passes,
        surface_equalizer_enabled=False,
        preserve_global_mean=True,
        debug_out=debug,
    )
    out, _alpha = y_cbcr_to_rgb_preserve_y(y2, c2)
    return torch.rot90(out, 3, dims=(-2, -1)), stats, debug

def main() -> int:
    args = parser().parse_args()
    root = args.input.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not (256 <= args.processing_size <= 2048) or args.processing_size % 8:
        raise ValueError("--processing-size must be a multiple of 8 in [256, 2048]")
    if not (0.0 <= args.second_pass_strength <= 2.0):
        raise ValueError("--second-pass-strength must be in [0, 2]")
    if not (0.0 <= args.first_pass_luma_strength <= 2.0):
        raise ValueError("--first-pass-luma-strength must be in [0, 2]")
    if not (0.0 <= args.first_pass_chroma_strength <= 2.0):
        raise ValueError("--first-pass-chroma-strength must be in [0, 2]")
    if not (0.0 <= args.tone_restore_strength <= 1.0):
        raise ValueError("--tone-restore-strength must be in [0, 1]")
    if not (1.0 < args.tone_restore_max_gain <= 4.0):
        raise ValueError("--tone-restore-max-gain must be in (1, 4]")
    if not (0.0 <= args.tone_restore_min_confidence <= 1.0):
        raise ValueError("--tone-restore-min-confidence must be in [0, 1]")
    if not (0.0 <= args.flat_cleanup_strength <= 3.0):
        raise ValueError("--flat-cleanup-strength must be in [0, 3]")
    if args.band_axis_auto_aspect_ratio <= 1.0:
        raise ValueError("--band-axis-auto-aspect-ratio must be > 1")
    if args.band_axis_analysis_size < 64:
        raise ValueError("--band-axis-analysis-size must be >= 64")
    if not (0 <= args.flat_local_edge_distance <= 100):
        raise ValueError("--flat-local-edge-distance must be in [0, 100]")
    if args.flat_profile_band_period != 0.0 and not (1.0 <= args.flat_profile_band_period <= 7680.0):
        raise ValueError("--flat-profile-band-period must be 0 (auto) or in [1, 7680]")
    if args.flat_local_correction_horizontal_sigma < 0:
        raise ValueError("--flat-local-correction-horizontal-sigma must be >= 0")
    if args.flat_local_application_horizontal_sigma < 0:
        raise ValueError("--flat-local-application-horizontal-sigma must be >= 0")
    if args.flat_profile_pwm_transition_ratio <= 0:
        raise ValueError("--flat-profile-pwm-transition-ratio must be > 0")
    if not (0.0 < args.flat_profile_pwm_min_duty < args.flat_profile_pwm_max_duty < 1.0):
        raise ValueError("PWM duty limits must satisfy 0 < min < max < 1")
    if args.flat_profile_pwm_min_transition_score <= 0:
        raise ValueError("--flat-profile-pwm-min-transition-score must be > 0")
    if args.flat_profile_adaptive_x_ratio <= 0 or args.flat_profile_adaptive_y_ratio <= 0:
        raise ValueError("--flat-profile-adaptive-x-ratio and --flat-profile-adaptive-y-ratio must be > 0")
    if not (0.0 <= args.flat_profile_adaptive_corr_low < args.flat_profile_adaptive_corr_high <= 1.0):
        raise ValueError("adaptive profile correlation thresholds must satisfy 0 <= low < high <= 1")
    if args.flat_profile_adaptive_max_gain <= 0:
        raise ValueError("--flat-profile-adaptive-max-gain must be > 0")
    if not (0.0 <= args.flat_profile_pwm_polish_strength <= 1.25):
        raise ValueError("--flat-profile-pwm-polish-strength must be in [0, 1.25]")
    if not (1 <= args.flat_profile_pwm_polish_passes <= 6):
        raise ValueError("--flat-profile-pwm-polish-passes must be in [1, 6]")
    if args.flat_broad_consensus_regions < 2:
        raise ValueError("--flat-broad-consensus-regions must be >= 2")
    if not (2 <= args.flat_broad_consensus_min_regions <= args.flat_broad_consensus_regions):
        raise ValueError("--flat-broad-consensus-min-regions must be between 2 and --flat-broad-consensus-regions")
    if not (0.0 <= args.flat_broad_consensus_corr_low < args.flat_broad_consensus_corr_high <= 1.0):
        raise ValueError("broad-consensus correlations must satisfy 0 <= low < high <= 1")
    if args.flat_broad_consensus_smooth_fraction <= 0 or args.flat_broad_consensus_baseline_fraction <= args.flat_broad_consensus_smooth_fraction:
        raise ValueError("broad-consensus fractions must satisfy 0 < smooth < baseline")
    device = choose_device(args.device)
    amp = bool(args.amp and device.type == "cuda")

    luma = None
    chroma = None
    if args.restormer:
        if args.luma_model is None or args.chroma_model is None:
            raise ValueError("--luma-model and --chroma-model are required when Restormer is enabled")
        luma = build_single_image_restormer()
        strict_load(luma, load_state_dict_file(args.luma_model.expanduser().resolve()))
        luma.to(device).eval()
        chroma = build_chroma_branch()
        strict_load(chroma, load_state_dict_file(args.chroma_model.expanduser().resolve()))
        chroma.to(device).eval()

    print(f"device: {device}; AMP: {amp}; restormer={args.restormer}; passes={args.passes}; exposure-lock={args.exposure_lock}; band-axis={args.band_axis}")
    print(
        f"luma-mode={args.luma_mode}; sigma-x={args.horizontal_sigma:g}; "
        f"row-anchor={not args.no_row_anchor}; flat-filter={args.flat_filter}; "
        f"flat-profile={args.flat_profile} mode={args.flat_profile_mode}; surface-equalizer={args.flat_surface_equalizer}"
        f"({args.flat_surface_equalizer_mode})"
    )
    images = discover(root)
    if not images:
        raise ValueError(f"No supported images found under {root}")

    written = 0
    for src in images:
        dst = out_for(src, root, out)
        if dst.exists() and not args.overwrite:
            print(f"skip  {dst}")
            continue
        with Image.open(src) as im:
            original = to_tensor(im)
        orthogonal_enabled = bool(args.orthogonal_profile or args.band_axis == "both")
        decision_request = "horizontal" if args.band_axis == "both" else args.band_axis
        axis_decision = decide_band_axis(
            original,
            requested=decision_request,
            portrait_ratio=args.band_axis_auto_aspect_ratio,
            analysis_size=args.band_axis_analysis_size,
        )
        axis_label = f"{axis_decision.axis} + orthogonal profile" if orthogonal_enabled else axis_decision.axis
        print(
            f"  axis: {axis_label} ({axis_decision.reason}); "
            f"scores H={axis_decision.horizontal_score:.5f} V={axis_decision.vertical_score:.5f}"
        )
        source_oriented = orient_for_processing(original.unsqueeze(0), axis_decision.axis).to(device)
        current = source_oriented.clone()

        infos = []
        domain = "log"
        if args.restormer:
            for pass_index in range(1, args.passes + 1):
                current, info, domain = run_pass(
                    current,
                    luma=luma,
                    chroma=chroma,
                    device=device,
                    amp=amp,
                    args=args,
                    pass_index=pass_index,
                )
                infos.append(info)
                print(
                    f"  pass {pass_index}: corr-rms={info.constrained_rms:.5f} "
                    f"exposure-removed={info.exposure_removed_stops:+.4f} stops "
                    f"gamut-compressed={100*info.gamut_compressed:.2f}%"
                )
        else:
            print("  Restormer disabled; using source image as deterministic-cleanup input")

        flat_stats = None
        flat_mask = None
        flat_debug = {}
        cleanup_enabled = bool(args.flat_filter or args.flat_profile or args.flat_surface_equalizer)
        if cleanup_enabled:
            final_y, final_c = rgb_to_y_cbcr(current.float())
            broad_neural_gain_hint = None
            broad_neural_chroma_hint = None
            cleanup_luma_hint = infos[-1].field if infos else None
            cleanup_chroma_hint = infos[-1].chroma_delta if infos else None
            needs_cumulative_hint = bool(
                (args.flat_profile and args.flat_profile_mode == "pwm")
                or (args.flat_surface_equalizer and args.flat_surface_equalizer_mode == "consensus")
            )
            if needs_cumulative_hint and args.restormer:
                source_y_for_cleanup, source_c_for_cleanup = rgb_to_y_cbcr(source_oriented.float())
                cumulative_luma_hint = torch.log(
                    (final_y.float().clamp_min(0.0) + float(args.luma_eps))
                    / (source_y_for_cleanup.float().clamp_min(0.0) + float(args.luma_eps))
                )
                cumulative_chroma_hint = final_c.float() - source_c_for_cleanup.float()
                # PWM timing must use the cumulative source -> post-neural
                # correction. With two passes, infos[-1] contains only pass 2,
                # which can be too weak to expose the repeated transition train.
                if args.flat_profile and args.flat_profile_mode == "pwm":
                    cleanup_luma_hint = cumulative_luma_hint
                    cleanup_chroma_hint = cumulative_chroma_hint
                if args.flat_surface_equalizer and args.flat_surface_equalizer_mode == "consensus":
                    broad_neural_gain_hint = cumulative_luma_hint
                    broad_neural_chroma_hint = cumulative_chroma_hint
            final_y, final_c, flat_mask, flat_stats = apply_flat_region_filter(
                final_y,
                final_c,
                luma_field=cleanup_luma_hint,
                chroma_delta_hint=cleanup_chroma_hint,
                band_period_px=(args.flat_band_period if args.flat_filter else 0.0),
                period_sigma_ratio=args.flat_period_sigma_ratio,
                coherence_sigma_x=args.flat_horizontal_sigma,
                flat_full=args.flat_full,
                flat_none=args.flat_none,
                luma_strength=(args.flat_luma_strength * args.flat_cleanup_strength if args.flat_filter else 0.0),
                chroma_strength=(args.flat_chroma_strength * args.flat_cleanup_strength if args.flat_filter else 0.0),
                luma_highpass=args.flat_highpass,
                luma_lowpass=args.flat_lowpass,
                shadow_ramp_lstar=args.flat_shadow_ramp,
                highlight_ramp_lstar=args.flat_highlight_ramp,
                luma_spatial_feather=args.flat_luma_spatial_feather,
                base_lstar_sigma_ratio=args.flat_base_lstar_sigma_ratio,
                edge_low=args.flat_edge_low,
                edge_high=args.flat_edge_high,
                edge_guard_px=args.flat_edge_guard,
                broad_structure_sigma=args.flat_broad_structure_sigma,
                broad_structure_low=args.flat_broad_structure_low,
                broad_structure_high=args.flat_broad_structure_high,
                broad_structure_guard_px=args.flat_broad_structure_guard,
                broad_structure_feather=args.flat_broad_structure_feather,
                coarse_preblur_sigma=args.flat_coarse_preblur,
                coarse_blend_weight=args.flat_coarse_blend,
                blend_feather=args.flat_blend_feather,
                local_extent_sigma=args.flat_local_extent_sigma,
                local_extent_low=args.flat_local_extent_low,
                local_extent_high=args.flat_local_extent_high,
                local_color_sigma=args.flat_local_color_sigma,
                local_color_preblur=args.flat_local_color_preblur,
                local_color_luma_tolerance=args.flat_local_color_luma_tolerance,
                local_color_chroma_tolerance=args.flat_local_color_chroma_tolerance,
                local_fill_sigma=args.flat_local_fill_sigma,
                local_fill_low=args.flat_local_fill_low,
                local_fill_high=args.flat_local_fill_high,
                local_edge_distance=args.flat_local_edge_distance,
                local_edge_feather=args.flat_local_edge_feather,
                # The 128 px processing-X correction regularizer was introduced
                # for horizontal displayed bands, where it suppresses support-hole
                # silhouettes across the row.  With vertical displayed bands the
                # image is rotated first, so the same operation maps back to a
                # long displayed-Y smear and can turn ceiling lights into vertical
                # streaks.  Bypass only this estimate regularizer for that axis;
                # the post-blend application regularizer remains active.
                local_correction_horizontal_sigma=(
                    0.0 if axis_decision.axis == "vertical"
                    else args.flat_local_correction_horizontal_sigma
                ),
                local_application_horizontal_sigma=args.flat_local_application_horizontal_sigma,
                residual_profile_luma_strength=(args.flat_profile_luma_strength if args.flat_profile else 0.0),
                residual_profile_chroma_strength=(args.flat_profile_chroma_strength if args.flat_profile else 0.0),
                residual_profile_mode=args.flat_profile_mode,
                residual_profile_narrow_ratio=args.flat_profile_narrow_ratio,
                residual_profile_pwm_transition_ratio=args.flat_profile_pwm_transition_ratio,
                residual_profile_pwm_min_duty=args.flat_profile_pwm_min_duty,
                residual_profile_pwm_max_duty=args.flat_profile_pwm_max_duty,
                residual_profile_pwm_min_transition_score=args.flat_profile_pwm_min_transition_score,
                residual_profile_base_ratio=args.flat_profile_base_ratio,
                residual_profile_band_period_px=args.flat_profile_band_period,
                period_mode=args.flat_profile_period_mode,
                period_min_px=args.flat_profile_period_min,
                period_max_fraction=args.flat_profile_period_max_fraction,
                period_analysis_h=args.flat_profile_period_analysis,
                residual_profile_huber_k=args.flat_profile_huber_k,
                residual_profile_min_coverage=args.flat_profile_min_coverage,
                residual_profile_adaptive=args.flat_profile_adaptive,
                residual_profile_adaptive_x_ratio=args.flat_profile_adaptive_x_ratio,
                residual_profile_adaptive_y_ratio=args.flat_profile_adaptive_y_ratio,
                residual_profile_adaptive_corr_low=args.flat_profile_adaptive_corr_low,
                residual_profile_adaptive_corr_high=args.flat_profile_adaptive_corr_high,
                residual_profile_adaptive_max_gain=args.flat_profile_adaptive_max_gain,
                residual_profile_no_harm=args.flat_profile_no_harm,
                residual_profile_pwm_polish=args.flat_profile_pwm_polish,
                residual_profile_pwm_polish_strength=args.flat_profile_pwm_polish_strength,
                residual_profile_pwm_polish_passes=args.flat_profile_pwm_polish_passes,
                surface_equalizer_enabled=args.flat_surface_equalizer,
                surface_equalizer_mode=args.flat_surface_equalizer_mode,
                surface_equalizer_luma_strength=args.flat_surface_equalizer_luma_strength,
                surface_equalizer_chroma_strength=args.flat_surface_equalizer_chroma_strength,
                surface_equalizer_poly_degree=args.flat_surface_equalizer_degree,
                surface_equalizer_preblur_sigma=args.flat_surface_equalizer_preblur,
                surface_equalizer_analysis_size=args.flat_surface_equalizer_analysis,
                surface_equalizer_component_threshold=args.flat_surface_equalizer_threshold,
                surface_equalizer_close_radius=args.flat_surface_equalizer_close_radius,
                surface_equalizer_min_area_fraction=args.flat_surface_equalizer_min_area,
                surface_equalizer_chroma_tolerance=args.flat_surface_equalizer_chroma_tolerance,
                surface_equalizer_luma_tolerance=args.flat_surface_equalizer_luma_tolerance,
                surface_equalizer_row_edge_barrier=args.flat_surface_equalizer_row_edge_barrier,
                surface_equalizer_row_edge_guard_px=args.flat_surface_equalizer_row_edge_guard,
                surface_equalizer_feather_sigma=args.flat_surface_equalizer_feather,
                surface_equalizer_row_sigma=args.flat_surface_equalizer_row_sigma,
                surface_equalizer_huber_k=args.flat_surface_equalizer_huber_k,
                surface_equalizer_min_coverage=args.flat_surface_equalizer_min_coverage,
                broad_consensus_regions=args.flat_broad_consensus_regions,
                broad_consensus_min_regions=args.flat_broad_consensus_min_regions,
                broad_consensus_corr_low=args.flat_broad_consensus_corr_low,
                broad_consensus_corr_high=args.flat_broad_consensus_corr_high,
                broad_consensus_smooth_fraction=args.flat_broad_consensus_smooth_fraction,
                broad_consensus_baseline_fraction=args.flat_broad_consensus_baseline_fraction,
                broad_neural_gain_hint=broad_neural_gain_hint,
                broad_neural_chroma_hint=broad_neural_chroma_hint,
                preserve_global_mean=not args.flat_allow_mean_shift,
                debug_out=flat_debug,
            )
            current, gamut_alpha = y_cbcr_to_rgb_preserve_y(final_y, final_c)
            compressed = float((gamut_alpha < 0.999).float().mean())
            lo_txt = "off" if flat_stats.luma_min_lstar is None else f"{flat_stats.luma_min_lstar:.2f}"
            hi_txt = "off" if flat_stats.luma_max_lstar is None else f"{flat_stats.luma_max_lstar:.2f}"
            print(
                f"  cleanup: local={args.flat_filter} profile={args.flat_profile} ({args.flat_profile_mode}) "
                f"surface-eq={args.flat_surface_equalizer} mode={args.flat_surface_equalizer_mode}; "
                f"blend>0.5={100*flat_stats.flat_fraction:.1f}% "
                f"support>0.5={100*flat_stats.support_fraction:.1f}% coarse>0.5={100*flat_stats.coarse_fraction:.1f}% "
                f"tone>0.5={100*flat_stats.luma_gate_fraction:.1f}% Lab-L*={lo_txt}..{hi_txt} "
                f"local-period={flat_stats.band_period_px:.1f}px sigma-y={flat_stats.sigma_y_px:.1f}px "
                f"Ydelta={flat_stats.y_delta_rms:.5f} Cdelta={flat_stats.c_delta_rms:.5f} "
                f"profile-period={flat_stats.profile_period_px:.1f}px conf={flat_stats.band_confidence:.2f} "
                f"profile-support>0.5={100*flat_stats.profile_support_fraction:.1f}% "
                f"profile-apply>0.5={100*flat_stats.profile_apply_fraction:.1f}% "
                f"profileY={flat_stats.profile_y_rms:.5f} profileC={flat_stats.profile_c_rms:.5f} "
                f"extent>0.5={100*flat_stats.local_extent_fraction:.1f}% local-safe>0.5={100*flat_stats.local_surface_fraction:.1f}% "
                f"surface-eq>0.5={100*flat_stats.surface_equalizer_fraction:.1f}% "
                f"eqY={flat_stats.surface_equalizer_y_rms:.5f} eqC={flat_stats.surface_equalizer_c_rms:.5f} "
                f"gamut-compressed={100*compressed:.2f}%"
            )
            if args.flat_surface_equalizer and args.flat_surface_equalizer_mode == "consensus":
                print(
                    f"    broad consensus: confidence={flat_stats.broad_consensus_confidence:.2f} "
                    f"agreeing-regions={flat_stats.broad_consensus_regions} "
                    f"Y={flat_stats.surface_equalizer_y_rms:.5f} "
                    f"C={flat_stats.surface_equalizer_c_rms:.5f}"
                )
            if args.flat_profile and flat_stats.band_candidates:
                cand = ", ".join(f"{p:.0f}px:{score:.2f}" for p, score in flat_stats.band_candidates)
                print(f"    profile period candidates: {cand}")

        orth_stats = None
        orth_debug = {}
        if orthogonal_enabled:
            current, orth_stats, orth_debug = run_orthogonal_profile_cleanup(current, args=args)
            print(
                f"  orthogonal-profile: period={orth_stats.profile_period_px:.1f}px "
                f"conf={orth_stats.band_confidence:.2f} "
                f"support>0.5={100*orth_stats.profile_support_fraction:.1f}% "
                f"apply>0.5={100*orth_stats.profile_apply_fraction:.1f}% "
                f"Y={orth_stats.profile_y_rms:.5f} C={orth_stats.profile_c_rms:.5f}"
            )


        # P11: restore global contrast/gamma. Applied as a smooth additive
        # offset in log space, fitted on band-axis-smoothed envelopes, so the
        # band residual passes through unchanged. Skipped when the period is
        # absent or unreliable -- band-axis smoothing is meaningless then.
        if args.tone_restore and args.tone_restore_strength > 0:
            _tone_conf = 0.0 if flat_stats is None else float(flat_stats.band_confidence)
            # P11b: pick a period we actually trust, else fall back, else 0.0.
            # band_confidence is 0.00 whenever --flat-profile is off, which is
            # an ABSENT measurement rather than a bad one -- the old gate
            # rejected those images outright even though period=0 simply
            # disables band smoothing and is perfectly safe.
            _tone_period = 0.0
            if flat_stats is not None:
                if _tone_conf >= args.tone_restore_min_confidence:
                    _tone_period = float(flat_stats.profile_period_px)
                elif float(flat_stats.band_period_px) > 3.0:
                    # Local flat stage's period: a measurement, not a
                    # placeholder. test1_input reports 95.3px against a true
                    # 96.3; test3_input_crop 39.0 against 38.9.
                    _tone_period = float(flat_stats.band_period_px)
            if True:
                _proc_y, _proc_c = rgb_to_y_cbcr(current.float())
                _ref_y, _ = rgb_to_y_cbcr(source_oriented.float())
                _new_y, _tone_stats = match_tone_log_torch(
                    _proc_y, _ref_y,
                    period=_tone_period,
                    strength=args.tone_restore_strength,
                    max_gain=args.tone_restore_max_gain,
                )
                current, _tone_alpha = y_cbcr_to_rgb_preserve_y(_new_y, _proc_c)
                print(
                    f"  tone-restore: strength={args.tone_restore_strength:.2f} "
                    f"period={_tone_period:.1f}px conf={_tone_conf:.2f} "
                    f"{'(no band smoothing)' if _tone_period <= 3.0 else ''} "
                    f"contrast={_tone_stats['contrast_ratio']:.3f} "
                    f"delta-max={_tone_stats['delta_max']:.4f} "
                    f"gamut-compressed={100*float((_tone_alpha < 0.999).float().mean()):.2f}%"
                )

        display_current = restore_display_orientation(current, axis_decision.axis)
        save_rgb(display_current.squeeze(0), dst)
        if args.debug_dir is not None:
            dbg = args.debug_dir.expanduser().resolve()
            stem = src.stem
            for i, info in enumerate(infos, 1):
                dbg_field = restore_display_orientation(info.field[0], axis_decision.axis)
                save_field_map(dbg_field, dbg / f"{stem}_pass{i}_luma_correction.png", domain)
            if flat_mask is not None:
                save_gray(restore_display_orientation(flat_mask[0], axis_decision.axis), dbg / f"{stem}_flat_blend_mask.png")
            for name in ("fine_mask", "coarse_mask", "local_extent_gate", "local_color_gate", "local_fill_gate", "local_edge_distance_gate", "local_surface_gate", "local_safe_gate", "local_structure_gate", "support_mask", "profile_support_mask", "profile_application_gate", "profile_confidence", "profile_apply_mask", "profile_adaptive_gain_y", "profile_adaptive_gain_c", "profile_adaptive_evidence_y", "profile_adaptive_evidence_c", "period_support", "edge_support", "broad_structure_support", "luma_gate", "raw_tone_support", "raw_tone_veto", "base_lstar", "surface_equalizer_candidate", "surface_equalizer_region", "surface_equalizer_apply", "broad_consensus_evidence", "broad_consensus_chroma_evidence", "broad_consensus_confidence"):
                if name in flat_debug:
                    save_gray(restore_display_orientation(flat_debug[name][0], axis_decision.axis), dbg / f"{stem}_flat_{name}.png")
            if orth_debug:
                for name in ("profile_support_mask", "profile_application_gate", "profile_confidence", "profile_apply_mask", "profile_adaptive_gain_y", "profile_adaptive_gain_c", "profile_adaptive_evidence_y", "profile_adaptive_evidence_c", "period_support"):
                    if name in orth_debug:
                        save_gray(torch.rot90(orth_debug[name][0], 3, dims=(-2, -1)), dbg / f"{stem}_orthogonal_{name}.png")
            if flat_stats is not None:
                lines = [
                    f"band_axis={axis_decision.axis}",
                    f"band_axis_reason={axis_decision.reason}",
                    f"band_axis_horizontal_score={axis_decision.horizontal_score:.6f}",
                    f"band_axis_vertical_score={axis_decision.vertical_score:.6f}",
                    f"local_period_px={flat_stats.band_period_px:.6f}",
                    f"profile_period_px={flat_stats.profile_period_px:.6f}",
                    f"profile_period_confidence={flat_stats.band_confidence:.6f}",
                    f"profile_support_fraction={flat_stats.profile_support_fraction:.6f}",
                    f"profile_apply_fraction={flat_stats.profile_apply_fraction:.6f}",
                    f"profile_confidence_mean={flat_stats.profile_confidence_mean:.6f}",
                    f"local_extent_fraction={flat_stats.local_extent_fraction:.6f}",
                    f"local_color_fraction={flat_stats.local_color_fraction:.6f}",
                    f"local_fill_fraction={flat_stats.local_fill_fraction:.6f}",
                    f"local_edge_distance_fraction={flat_stats.local_edge_distance_fraction:.6f}",
                    f"local_surface_fraction={flat_stats.local_surface_fraction:.6f}",
                    f"surface_equalizer_fraction={flat_stats.surface_equalizer_fraction:.6f}",
                    f"surface_equalizer_y_rms={flat_stats.surface_equalizer_y_rms:.6f}",
                    f"surface_equalizer_c_rms={flat_stats.surface_equalizer_c_rms:.6f}",
                    f"surface_equalizer_mode={args.flat_surface_equalizer_mode}",
                    f"broad_consensus_confidence={flat_stats.broad_consensus_confidence:.6f}",
                    f"broad_consensus_regions={flat_stats.broad_consensus_regions}",
                    "candidates=" + ", ".join(f"{p:.3f}px:{score:.6f}" for p, score in flat_stats.band_candidates),
                ]
                (dbg / f"{stem}_band_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            (dbg / f"{stem}_band_axis.txt").write_text(
                "\n".join([
                    f"axis={axis_label}",
                    f"reason={axis_decision.reason}",
                    f"horizontal_score={axis_decision.horizontal_score:.6f}",
                    f"vertical_score={axis_decision.vertical_score:.6f}",
                    f"aspect_ratio={axis_decision.aspect_ratio:.6f}",
                ]) + "\n", encoding="utf-8"
            )
        print(f"write {dst}")
        written += 1

    print(f"done: wrote {written} image(s); discovered {len(images)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
