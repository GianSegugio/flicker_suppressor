from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from band_axis import decide_band_axis, orient_for_processing, restore_display_orientation
from chroma_branch_model import build_chroma_branch
from color_space import rgb_to_y_cbcr, y_cbcr_to_rgb_preserve_y
from flat_region_filter import apply_flat_region_filter
from hybrid_infer_detail_preserving import run_pass, run_orthogonal_profile_cleanup, save_rgb, to_tensor
from restormer_model import build_single_image_restormer, choose_device, load_state_dict_file, strict_load
from .settings_schema import namespace


def _load_stage_mask(path: Path, target_hw: tuple[int,int]) -> torch.Tensor:
    """Load a GUI paint mask as 1x1xHxW alpha in display orientation."""
    h,w=(int(target_hw[0]),int(target_hw[1]))
    with Image.open(path) as im:
        rgba=im.convert('RGBA')
        alpha=rgba.getchannel('A')
        if alpha.size!=(w,h):
            alpha=alpha.resize((w,h),Image.Resampling.BILINEAR)
        raw=bytearray(alpha.tobytes())
    values=torch.frombuffer(raw,dtype=torch.uint8).clone().view(1,1,h,w).float()/255.0
    return values.clamp(0.0,1.0)


class CancelledError(RuntimeError):
    pass


class FlickerEngine:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._luma = None
        self._chroma = None
        self._device = None
        self._lock = threading.RLock()

    def _check(self, ev):
        if ev is not None and ev.is_set():
            raise CancelledError("Operation cancelled")

    def _log(self, cb, text):
        if cb:
            cb(text)

    def _ensure_models(self, requested_device: str, cb=None):
        device = choose_device(requested_device)
        if self._luma is None:
            self._log(cb, "Loading neural models...")
            self._luma = build_single_image_restormer()
            strict_load(self._luma, load_state_dict_file(self.models_dir / "y.pth"))
            self._chroma = build_chroma_branch()
            strict_load(self._chroma, load_state_dict_file(self.models_dir / "chroma.pth"))
        if self._device != device:
            self._luma.to(device).eval()
            self._chroma.to(device).eval()
            self._device = device
        return device

    def process_file(self, src: Path, dst: Path, settings: dict, masks: dict[str,Path]|None=None, callback: Callable[[str], None] | None = None, cancel_event=None):
        args = namespace(settings)
        with self._lock:
            self._check(cancel_event)
            if args.restormer:
                device = self._ensure_models(str(args.device), callback)
            else:
                device = choose_device(str(args.device))
                self._log(callback, "Restormer disabled")
            amp = bool(args.amp and device.type == "cuda")
            with Image.open(src) as im:
                original = to_tensor(im)
            orthogonal_enabled = bool(args.orthogonal_profile or args.band_axis == "both")
            decision_request = "horizontal" if args.band_axis == "both" else args.band_axis
            axis = decide_band_axis(original, requested=decision_request, portrait_ratio=args.band_axis_auto_aspect_ratio, analysis_size=args.band_axis_analysis_size)
            self._log(callback, f"Band direction: {axis.axis} ({axis.reason})")
            source_oriented = orient_for_processing(original.unsqueeze(0), axis.axis).to(device)
            stage_masks={}
            for stage,path in (masks or {}).items():
                if stage not in {'flat','profile','broad'}:
                    continue
                p=Path(path)
                if not p.exists():
                    continue
                display_mask=_load_stage_mask(p,original.shape[-2:])
                stage_masks[stage]=orient_for_processing(display_mask,axis.axis).to(device)
            current = source_oriented.clone()
            infos = []
            if args.restormer:
                for pass_index in range(1, args.passes + 1):
                    self._check(cancel_event)
                    self._log(callback, f"Neural pass {pass_index}/{args.passes}")
                    current, info, _domain = run_pass(current, luma=self._luma, chroma=self._chroma, device=device, amp=amp, args=args, pass_index=pass_index)
                    infos.append(info)

            flat_stats = None
            cleanup_enabled = bool(args.flat_filter or args.flat_profile or args.flat_surface_equalizer)
            if cleanup_enabled:
                self._check(cancel_event)
                self._log(callback, "Residual cleanup")
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
                    if args.flat_profile and args.flat_profile_mode == "pwm":
                        cleanup_luma_hint = cumulative_luma_hint
                        cleanup_chroma_hint = cumulative_chroma_hint
                    if args.flat_surface_equalizer and args.flat_surface_equalizer_mode == "consensus":
                        broad_neural_gain_hint = cumulative_luma_hint
                        broad_neural_chroma_hint = cumulative_chroma_hint
                final_y, final_c, _mask, flat_stats = apply_flat_region_filter(
                    final_y, final_c,
                    luma_field=cleanup_luma_hint, chroma_delta_hint=cleanup_chroma_hint,
                    band_period_px=(args.flat_band_period if args.flat_filter else 0.0),
                    period_sigma_ratio=args.flat_period_sigma_ratio,
                    coherence_sigma_x=args.flat_horizontal_sigma,
                    flat_full=args.flat_full, flat_none=args.flat_none,
                    luma_strength=(args.flat_luma_strength * args.flat_cleanup_strength if args.flat_filter else 0.0),
                    chroma_strength=(args.flat_chroma_strength * args.flat_cleanup_strength if args.flat_filter else 0.0),
                    luma_highpass=args.flat_highpass, luma_lowpass=args.flat_lowpass,
                    shadow_ramp_lstar=args.flat_shadow_ramp, highlight_ramp_lstar=args.flat_highlight_ramp,
                    luma_spatial_feather=args.flat_luma_spatial_feather,
                    base_lstar_sigma_ratio=args.flat_base_lstar_sigma_ratio,
                    edge_low=args.flat_edge_low, edge_high=args.flat_edge_high, edge_guard_px=args.flat_edge_guard,
                    broad_structure_sigma=args.flat_broad_structure_sigma, broad_structure_low=args.flat_broad_structure_low,
                    broad_structure_high=args.flat_broad_structure_high, broad_structure_guard_px=args.flat_broad_structure_guard,
                    broad_structure_feather=args.flat_broad_structure_feather,
                    coarse_preblur_sigma=args.flat_coarse_preblur, coarse_blend_weight=args.flat_coarse_blend,
                    blend_feather=args.flat_blend_feather,
                    local_extent_sigma=args.flat_local_extent_sigma, local_extent_low=args.flat_local_extent_low, local_extent_high=args.flat_local_extent_high,
                    local_color_sigma=args.flat_local_color_sigma, local_color_preblur=args.flat_local_color_preblur,
                    local_color_luma_tolerance=args.flat_local_color_luma_tolerance, local_color_chroma_tolerance=args.flat_local_color_chroma_tolerance,
                    local_fill_sigma=args.flat_local_fill_sigma, local_fill_low=args.flat_local_fill_low, local_fill_high=args.flat_local_fill_high,
                    local_edge_distance=args.flat_local_edge_distance, local_edge_feather=args.flat_local_edge_feather,
                    local_correction_horizontal_sigma=(
                        0.0 if axis.axis == "vertical"
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
                    local_user_mask=stage_masks.get('flat'),
                    profile_user_mask=stage_masks.get('profile'),
                    surface_user_mask=stage_masks.get('broad'),
                    preserve_global_mean=not args.flat_allow_mean_shift,
                    debug_out=None,
                )
                current, _alpha = y_cbcr_to_rgb_preserve_y(final_y, final_c)

            if orthogonal_enabled:
                self._check(cancel_event)
                self._log(callback, "Orthogonal cleanup")
                before_orthogonal=current
                orthogonal_out, _stats, _debug = run_orthogonal_profile_cleanup(current, args=args)
                profile_mask=stage_masks.get('profile')
                if profile_mask is not None:
                    # The Residual profile section mask owns both the primary
                    # profile and its optional orthogonal pass. Blend only the
                    # final orthogonal delta so the perpendicular estimator can
                    # still use the whole image as evidence.
                    current=before_orthogonal + profile_mask * (orthogonal_out-before_orthogonal)
                else:
                    current=orthogonal_out

            self._check(cancel_event)
            self._check(cancel_event)
            display = restore_display_orientation(current, axis.axis)
            save_rgb(display.squeeze(0), Path(dst))
            self._log(callback, "Done")
            return {"axis": axis.axis, "profile_period": getattr(flat_stats, "profile_period_px", None)}
