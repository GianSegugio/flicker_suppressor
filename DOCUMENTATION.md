# Flicker Suppressor - Architecture, GUI, and Command Reference

This document describes the release architecture in depth, documents every option accepted by `hybrid_infer_detail_preserving.py`, and records the current desktop GUI workflow and processing policy.

## 1. Problem definition

Rolling-shutter flicker occurs when a sensor exposes/readouts different rows at different times while an artificial light source changes intensity or spectral balance during the frame. In a displayed image the artifact usually appears as broad or fine bands. Depending on camera orientation, the visible bands may be horizontal or vertical.

Flicker Suppressor treats the problem as a combination of three separable tasks:

1. **luminance restoration** - estimate the illumination correction without replacing fine scene detail;
2. **chrominance restoration** - remove row-dependent Cb/Cr casts without letting the chroma branch alter luminance;
3. **deterministic residual cleanup** - optionally suppress remaining band structure using geometry-aware image processing rather than asking the neural network to reconstruct the whole image again.

The system is intentionally modular. A difficult case can receive stronger residual cleanup without retraining either neural model.

---

## 2. High-level architecture

```text
                         INPUT RGB
                             |
                             v
                 EXIF orientation normalization
                             |
                             v
                  band-axis decision/override
                             |
             +---------------+---------------+
             |                               |
       horizontal bands                vertical bands
       process directly            rotate 90 degrees
             |                               |
             +---------------+---------------+
                             |
                             v
                    processing RGB image
                             |
                 resize to 512 x 512 by default
                             |
              +--------------+---------------+
              |                              |
              v                              v
       RGB Restormer Y branch        2-channel CbCr branch
              |                              |
       candidate restored Y          candidate restored Cb/Cr
              |                              |
       derive correction field       derive Delta Cb / Delta Cr
              |                              |
       directional constraint                |
              |                              |
              +--------------+---------------+
                             |
                 apply corrections to ORIGINAL
                 full-resolution Y/Cb/Cr data
                             |
                 gamut-safe YCbCr -> RGB
                             |
                     optional second pass
                             |
                      optional cleanup
        +--------------------+---------------------+
        |                    |                     |
  local flat filter    robust row profile   large-surface equalizer
        |                    |                     |
        +--------------------+---------------------+
                             |
                optional orthogonal profile
                 (`--orthogonal-profile`)
                             |
                selective highlight recovery
                             |
                    restore orientation
                             |
                          PNG output
```

The neural networks operate at the configurable processing resolution. Their output is primarily treated as an **estimate of correction**, which is then resized and applied to the original-resolution image. This is a key design choice for preserving detail.

In the desktop GUI, Flat-region cleanup, Residual profile, and Broad residual cleanup can each have an optional painted mask. These masks are **final application gates**, not alternate estimators: the automatic stage can still analyze the complete image, but its visible delta is multiplied by the user mask before it reaches the output. This keeps period/surface estimation stable while guaranteeing that authored masks restrict where the corresponding cleanup is shown.

---

## 3. Neural model architecture

Both branches use the reduced Restormer configuration inherited from the BurstDeflicker baseline:

```text
base dimension:          32
encoder/latent blocks:   (2, 3, 4, 5)
attention heads:         (1, 2, 4, 8)
refinement blocks:       2
FFN expansion factor:    2.66
LayerNorm:               WithBias
input channels:          3
```

### 3.1 Luminance branch - `models/y.pth`

The luminance checkpoint is still a 3-channel RGB Restormer. Flicker Suppressor does **not** simply accept its final RGB image. Instead:

1. the model predicts a candidate restored RGB image at the processing resolution;
2. that prediction is converted to YCbCr;
3. only its **Y** channel is used;
4. a luminance correction field is computed between input Y and predicted Y;
5. the correction field is directionally constrained;
6. that field is applied to the original full-resolution Y.

This allows the neural model to estimate the flicker correction while the original image supplies the high-frequency scene detail.

### 3.2 Chroma branch - `models/chroma.pth`

The dedicated chroma branch shares the Restormer trunk but has a **two-channel output head**. Its outputs are centered Cb and Cr, not RGB.

At inference:

```text
Delta CbCr = predicted CbCr - input CbCr
```

The delta is resized to full resolution and added to the original Cb/Cr. Because this branch never outputs Y, it has no direct path to brighten or darken the final image.

### 3.3 Why the branches are separate

Earlier unified RGB experiments showed two recurring problems:

- a chroma-specialized RGB model could overcorrect luminance;
- a luminance band could trigger a false blue/cyan chroma correction, even in grayscale images.

Separating Y and CbCr prevents the first failure structurally and greatly reduces the second.

---

## 4. Color representation and gamut-safe recombination

Flicker Suppressor uses full-range BT.601-style YCbCr math:

```text
Y  =  0.299000 R + 0.587000 G + 0.114000 B
Cb = -0.168736 R - 0.331264 G + 0.500000 B
Cr =  0.500000 R - 0.418688 G - 0.081312 B
```

Cb and Cr are stored zero-centered, so neutral gray has Cb=Cr=0.

A corrected Y/Cb/Cr triplet can sometimes imply RGB values outside `[0,1]`. Naively clipping RGB would change luminance. Flicker Suppressor therefore uses a gamut-safe conversion that reduces chroma toward neutral only where needed while preserving the requested Y as closely as numerical precision allows.

The console reports the percentage of pixels that needed this chroma compression as `gamut-compressed=...%`.

---

## 5. Detail-preserving luminance correction

The default `--luma-mode directional` uses a stabilized log-gain field:

```text
g_raw = log((Y_pred + eps) / (Y_input + eps))
```

with `eps=0.02` by default. The field is clipped to a maximum number of exposure stops, then horizontally smoothed. The default horizontal smoothing sigma is 16 model-resolution pixels.

The physical assumption is that rolling-shutter bands are coherent across much of each sensor row, whereas real scene texture varies locally across X. Smoothing the **correction field**, instead of smoothing the image, removes texture contamination from the network correction.

The filtered gain is resized to the original image size and applied as:

```text
Y_out = (Y_original + eps) * exp(strength * g_filtered) - eps
```

### Row anchoring

With the default row anchor, horizontal filtering preserves the mean correction of each row. This keeps the directional constraint from weakening the overall correction the neural model estimated for that row.

### Alternative Y modes

- `raw` - legacy additive neural correction with no directional constraint;
- `directional` - default log-gain correction, horizontally constrained;
- `directional-additive` - same idea but using additive Delta Y instead of log gain;
- `row` - one robust weighted correction value per row.

`directional` is the recommended general mode.

---

## 6. Recursive two-pass processing and exposure lock

`--passes 2` feeds the first restored result through the neural pipeline a second time. This is useful for unusually severe residual bands.

A naive second pass tends to accumulate a positive global luminance shift. The default `--exposure-lock pass2` removes the global DC component from the second Y correction before it is applied. The local row/band correction remains.

You can reduce pass-two aggressiveness with, for example:

```powershell
--passes 2 `
--second-pass-strength 0.70
```

The normal CLI default remains one pass.

The desktop GUI deliberately hides Exposure lock and forces `exposure_lock = all` in its complete inference namespace. This applies the global-DC lock to both passes and also prevents older/pasted GUI settings from restoring a different exposure-lock value. The CLI remains independently configurable and keeps its parser default of `pass2`.

### 6.1 Selective highlight recovery

The neural correction can occasionally pull very bright source highlights downward even when the global exposure/DC component is locked. Flicker Suppressor therefore performs a final selective luminance recovery after neural and deterministic cleanup.

The recovery compares the processed Y channel with the **original oriented source Y**. It only restores luminance that processing removed; it never darkens a pixel or replaces corrected chroma. A smooth source-luminance gate begins at Y `0.90` and reaches full weight at Y `0.99`, so shadows, midtones, and ordinary bright surfaces are unaffected. The default recovery strength is `1.0` (100%). Gamut-safe YCbCr recombination is used after recovery.

The GUI exposes only **Highlight recovery**. Advanced CLI users can also change the gate endpoints with `--highlight-recovery-start` and `--highlight-recovery-full`. Set `--highlight-recovery-strength 0` to reproduce the pipeline without this final recovery.

---

## 7. Band-axis normalization

The trained models and deterministic filters are fundamentally row-oriented: they expect visible horizontal bands that vary over Y.

### Auto mode

`--band-axis auto` uses image geometry as its primary physical prior:

- portrait H/W >= 1.10 -> treat visible bands as vertical and rotate internally;
- landscape H/W <= 1/1.10 -> native horizontal processing;
- near-square -> use a coarse row-vs-column striping score.

This is based on the idea that physically rotating the camera rotates sensor rows in the displayed image.

### Manual modes

```text
horizontal  process directly
vertical    rotate 90 degrees, process, rotate back
both        horizontal primary restoration + vertical profile-only cleanup
```

Use manual override when a file was cropped or rotated after capture and Auto chooses the wrong orientation.

Orthogonal cleanup is enabled explicitly with `--orthogonal-profile`. It does not run the neural models twice: it keeps the primary restoration and applies only the texture-preserving residual-profile algorithm in the perpendicular axis. The legacy `--band-axis both` CLI value remains as a compatibility shortcut for horizontal primary processing plus orthogonal cleanup.

---

## 8. Optional local flat-region filter

Enable with:

```powershell
--flat-filter
```

This filter does **not** paste a blurred wall over the image. It estimates a row-coherent correction and blends only that correction into safe regions.

The GUI exposes the two local correction strengths directly: **Flat luminance strength** defaults to `0.70` and **Flat chroma strength** defaults to `0.85`. The legacy CLI-only `--flat-cleanup-strength` multiplier remains at `1.0` for backward compatibility but is not a GUI control.

### 8.1 Fine/coarse surface masks

The filter estimates low-detail regions while discounting row-coherent structure that may itself be flicker. A fine mask protects texture; a coarse preblurred mask allows mildly textured but broadly uniform surfaces to participate.

### 8.2 Edge-conductive estimation and row-coherent correction regularization

Dark objects must not contaminate nearby wall estimates. The local stage therefore uses masked normalized estimation, so excluded pixels do not directly contribute to the correction field. Earlier versions still had an important weakness: a wide vertical Gaussian could *bridge across* excluded edge rows. A real horizontal wall moulding, furniture boundary, or other scene structure could therefore be treated as if it were residual flicker, especially when foreground people interrupted the surface.

The current local estimator is **edge-conductive along Y**. It uses the existing soft scene-edge support as a propagation conductivity: evidence travels normally inside a smooth surface, but is progressively attenuated when the estimator crosses a strong real scene boundary. This avoids both failure modes of the earlier approaches: ordinary masked Gaussian smoothing could bridge across real edges, while hard segmentation could create visible seams. The operation is applied only while estimating the correction field; the source image is never blurred.

After that edge-aware estimate, the local stage can horizontally regularize the estimated **correction field**. This follows the physical assumption that rolling-shutter band correction should be coherent across a substantial part of each sensor row. For normal displayed horizontal bands, the default correction-field sigma is 128 full-resolution pixels and can be disabled for regression comparison with `--flat-local-correction-horizontal-sigma 0`.

For **displayed vertical bands**, the image is rotated into the row-oriented processing domain. In that orientation the same 128 px processing-X regularizer would map back into long displayed-Y smearing and can turn bright ceiling lights or other elongated fixtures into visible vertical streaks. Flicker Suppressor therefore bypasses this first correction-field regularizer automatically when the resolved band direction is vertical. The later post-blend regularizer remains active.

The local safety mask also includes a broad **structure-density guard**. Long high-contrast fixtures, frames, moulding, and similar dense edge regions are softly suppressed from the visible local correction so fragmented flat support around those structures cannot imitate a band. Open flat surfaces remain eligible.

A second **broad/defocus structure guard** handles the opposite failure mode: real boundaries that are so out of focus that their per-pixel gradient is too weak for the ordinary edge detector. The image is inspected at a broader scale and the gradient is scale-normalized, so a blurred head, wall panel, or moulding transition can still block local flattening while faint post-neural rolling bands normally remain eligible. This guard affects only the local flat-region stage; it does not reduce the texture-tolerant residual-profile support.

A second horizontal regularization is applied **after the visible local blend mask has been multiplied into the correction**. This prevents a protected person/object from becoming an abrupt hole in a row-varying correction field. Its default sigma is 48 full-resolution pixels and it can be disabled with `--flat-local-application-horizontal-sigma 0`.

### 8.3 Local same-surface safety

To prevent smooth skin or small objects from being mistaken for a wall, local flattening also requires:

- large-surface extent;
- broad Lab-lightness/CbCr color consistency;
- a high large-window same-surface fill ratio;
- sufficient distance from strong scene/object edges.

These affect the **visible local blend**, not the global profile stage.

### 8.4 Tone gate

The default flat-filter window protects deep shadows and highlights using CIE Lab lightness equivalents:

```text
dark midpoint:   #232323
bright midpoint: #efefef
```

Use hashless values in PowerShell:

```powershell
--flat-highpass 232323 `
--flat-lowpass efefef
```

The transitions are smooth rather than hard. The base lightness estimate is also made band-resistant so a dark flicker trough does not automatically disqualify an otherwise valid wall pixel.

---

## 9. Robust residual profile filter

Enable with:

```powershell
--flat-profile
```

This stage is different from the local flattener. It is designed for faint globally coherent residual bands on **textured** surfaces. It can be enabled independently of `--flat-filter`; the two stages share tone/edge support machinery but apply different corrections.

It:

1. uses tone/edge-safe pixels without requiring them to be locally flat;
2. forms robust cross-column row measurements using Huber-like outlier rejection;
3. detects a band-period family with a multiscale/harmonic-aware spectrum;
4. separates narrow residual structure from slower baseline illumination;
5. keeps that period/phase/waveform global, but by default fits a **slowly varying local amplitude** for it;
6. applies the resulting Y/Cb/Cr correction without blurring scene texture.

The adaptive-amplitude step addresses scenes where the same flicker waveform is present at different strengths on foreground and background surfaces. It does **not** estimate an independent profile in each tile or object: the robust global period/phase/waveform remains the reference. The local multiplier is fitted from the **band-limited residual belonging to that same frequency family**, which prevents broad wall shading, faces, clothing, and other unrelated low-frequency scene structure from becoming patch-shaped gain islands.

Adaptive amplitude is on by default. The current defaults use a relatively broad 1.50-period vertical fit, require stronger local waveform correlation (`0.15` to begin, `0.45` for full evidence), and cap the local multiplier at `1.0`. Thus adaptive mode attenuates weakly supported areas by default but does not silently amplify the profile above the user's selected global strength. For regression comparison, `--no-flat-profile-adaptive` restores the previous single global profile amplitude.

A second **no-harm gate** is also enabled by default. It evaluates the actual proposed profile correction, including application mask and user-selected strength, in the same band-limited domain before and after correction. Unlike the earlier 2-D validator, the current gate is deliberately **band-coherent**: validation evidence is collapsed across the waveform axis, broadly smoothed only along the band direction, and then expanded back over the waveform axis. It can therefore attenuate a profile in a large scene zone where that waveform would make the residual worse, while being mathematically unable to trace a blurred person/object silhouette or modulate individual bright/dark bands. `--no-flat-profile-no-harm` is retained for regression comparison.

When **Residual profile** and **Flat-region cleanup** are enabled together, a final **dominant-surface handoff** prevents the two stages from fighting each other at high profile strength. Profile estimation, adaptive gain, exposure/DC anchoring, and no-harm validation are completed first. Only after those calculations, one large connected, color-consistent smooth surface is selected as the surface primarily owned by Flat-region cleanup. This avoids the earlier broadly blurred ownership mask, which could leak from a background wall into smooth foreground skin. The selected surface receives a lightly feathered cap on the *visible* profile contribution. The current internal cap is an effective profile strength of `0.50`; profile strengths at or below `0.50` are unchanged on that surface. Foreground/textured or differently colored smooth regions retain the user's full requested profile strength. The handoff is reduced proportionally if the local flat Y/C strength is deliberately set below its normal `0.70`/`0.85` authority, and it is disabled entirely when Flat-region cleanup is off. This lets a user drive Residual profile hard for banded skin/clothing without painting broad bright/dark profile lobes into smooth defocused walls.

For intentionally aggressive profile strengths above `1.0`, the handoff uses a **one-sided boundary transition** rather than an abrupt wall/foreground switch. The selected background surface is extended a short distance outward and smoothly released into the foreground. This protects the background immediately behind a defocused silhouette while returning to the user's full strength over the foreground interior, avoiding both the earlier background blobs and the later hard halo that appeared when the cap changed too sharply at a head/object boundary.

There are two additional high-strength safeguards. First, bright soft scene edges combine the ordinary edge support with the broad/defocus structure support; when Y or C profile strength is above `1.0`, the visible profile contribution around such bright boundaries is locally capped to an effective strength of `0.50`. Darker skin/face interiors are not targeted by this bright-edge gate. Second, positive luminance-profile corrections on already-bright detail are limited by the remaining highlight headroom, with the limiter fading in through the bright range. Negative corrections are left untouched. This reduces bright flowers, pale masks, specular details, and similar features being pushed into local glow/halo artifacts when Residual profile is deliberately driven hard.

Useful strengths:

```text
0.35  default conservative cleanup
0.50  strong residual cleanup
0.7-1.0 very strong
1.0-1.2 extreme, strongly periodic residuals
1.5-2.0 intentionally aggressive; useful only when the foreground needs it and should be inspected carefully
```

For a known fine period, for example 37 px:

```powershell
--flat-profile `
--flat-profile-band-period 37 `
--flat-profile-luma-strength 1.00 `
--flat-profile-chroma-strength 0.80
```

---

## 10. Dominant large-surface equalizer

Enable with:

```powershell
--flat-surface-equalizer
```

This is for a different regime: only a few very broad residual cycles remain on one large surface, and ordinary frequency separation cannot clearly distinguish flicker from real illumination.

The equalizer:

- finds one dominant large, color-consistent surface;
- estimates robust Y/Cb/Cr values per row;
- fits a low-order polynomial baseline (quadratic by default) representing legitimate illumination/color drift;
- treats departures from that baseline as residual flicker;
- applies only the row correction to the selected surface;
- feathers the surface boundary.

It is intentionally opt-in because it makes a stronger assumption than the normal profile filter. It can be enabled independently of both `--flat-filter` and `--flat-profile`.

---

### 10.1 Cleanup-stage independence

The three deterministic cleanup stages are independently switchable:

- `--flat-filter` enables the local flat-region correction only;
- `--flat-profile` enables the global robust residual row-profile correction only;
- `--flat-surface-equalizer` enables the dominant large-surface equalizer only.

They deliberately share tone/edge support calculations and can be combined, but none of these three switches is a prerequisite for the others. `--orthogonal-profile` is separate again: it adds an orthogonal profile pass after the primary-axis processing and reuses the residual-profile algorithm in the perpendicular direction.

When the main Residual profile and Flat-region cleanup are both enabled, the profile stage is applied **first**. The local flat-region correction is then estimated from what remains, so local support-mask artifacts are not fed back into the adaptive profile fit. The dominant large-surface equalizer, when enabled, runs after those two corrections.

GUI paint masks do not change that estimation order. The Residual mask gates the main profile and optional Orthogonal profile at application time; the Flat mask gates the local flat correction after its horizontal regularization and also limits where Flat-region cleanup may claim ownership for the profile handoff; the Broad mask gates only the final large-surface-equalizer delta after the equalizer has fitted its automatically selected surface.

---

## 11. Debug output

`--debug-dir PATH` writes visual diagnostics beside the normal output. Depending on enabled stages, files include:

```text
*_pass1_luma_correction.png
*_pass2_luma_correction.png
*_flat_blend_mask.png
*_flat_fine_mask.png
*_flat_coarse_mask.png
*_flat_local_extent_gate.png
*_flat_local_color_gate.png
*_flat_local_fill_gate.png
*_flat_local_edge_distance_gate.png
*_flat_local_surface_gate.png
*_flat_local_safe_gate.png
*_flat_local_structure_gate.png
*_flat_support_mask.png
*_flat_profile_support_mask.png
*_flat_profile_application_gate.png
*_flat_profile_confidence.png
*_flat_profile_apply_mask.png
*_flat_profile_adaptive_gain_y.png
*_flat_profile_adaptive_gain_c.png
*_flat_profile_adaptive_evidence_y.png
*_flat_profile_adaptive_evidence_c.png
*_flat_period_support.png
*_flat_edge_support.png
*_flat_broad_structure_support.png
*_flat_luma_gate.png
*_flat_raw_tone_support.png
*_flat_raw_tone_veto.png
*_flat_base_lstar.png
*_flat_surface_equalizer_candidate.png
*_flat_surface_equalizer_region.png
*_flat_surface_equalizer_apply.png
*_orthogonal_profile_support_mask.png      # when --orthogonal-profile
*_orthogonal_profile_application_gate.png
*_orthogonal_profile_confidence.png
*_orthogonal_profile_apply_mask.png
*_orthogonal_profile_adaptive_gain_y.png
*_orthogonal_profile_adaptive_gain_c.png
*_orthogonal_profile_adaptive_evidence_y.png
*_orthogonal_profile_adaptive_evidence_c.png
```

The debug directory also receives a text summary containing band-axis and period-analysis information.

---

## 12. Practical tuning recipes

Start with the least aggressive recipe that matches the visible problem, inspect the result, and increase strength only if residual bands remain.

Unless noted otherwise, the snippets below are additional options that can be appended to the normal inference command from section 12.1.

### 12.1 Normal restoration

Use this first for an ordinary image with rolling-shutter flicker. One neural pass is the safest default and is often sufficient.

```powershell
python .\hybrid_infer_detail_preserving.py `
    --input .\photo.jpg `
    --output .\photo_restored.png `
    --luma-model .\models\y.pth `
    --chroma-model .\models\chroma.pth `
    --device cuda `
    --amp
```

If the result is already satisfactory, do not enable additional cleanup stages merely because they are available.

### 12.2 Severe bands that remain after one pass

Use this when the first neural pass clearly reduces the artifact but broad or high-contrast bands are still visible. A second pass can remove more of the residual correction, while the default pass-two exposure lock limits recursive global brightening. The local flat filter is useful when much of the residual is visible over broad, relatively simple surfaces such as painted walls or ceilings.

```powershell
--passes 2 `
--flat-filter
```

If the second pass is too aggressive, reduce it rather than disabling exposure protection:

```powershell
--passes 2 `
--second-pass-strength 0.70 `
--flat-filter
```

### 12.3 Faint residual bands on textured or mostly uniform surfaces

Use this when the neural restoration is already good, but low-amplitude periodic bands remain visible over surfaces that still contain texture, grain, plaster detail, fabric structure, or other fine information that should not be blurred.

The residual profile stage applies a one-dimensional row/column correction rather than spatially blurring the image, so it is generally safer for fine detail than simply increasing the local flat-filter strength.

```powershell
--flat-profile `
--flat-profile-luma-strength 0.50 `
--flat-profile-chroma-strength 0.50
```

If only luminance bands remain, increase the luma profile before the chroma profile. If only colored bands remain, increase chroma more cautiously.

### 12.4 Fine, closely spaced, high-contrast periodic bands

Use this when the remaining bands repeat at a short, regular interval and are still very visible after normal cleanup. This regime often benefits more from a stronger residual profile than from a stronger local flattener.

A useful starting point is:

```powershell
--passes 2 `
--flat-profile `
--flat-profile-luma-strength 1.00 `
--flat-profile-chroma-strength 0.80
```

For extreme periodic residuals, luma strengths around `1.0-1.2` can be useful. Values above `1.0` intentionally over-apply the estimated residual correction, so increase gradually and inspect the result.

If the band period is known, it can be forced. For example, if diagnostics show approximately 37 pixels between repeating bands:

```powershell
--flat-profile-band-period 37
```

`37` is only an example. Do **not** use it for every image. If the period is unknown, leave the option at its default `0` so the multiscale detector estimates it automatically. `--debug-dir` can be used to inspect the selected period and candidate family.

### 12.5 Protect shadows and highlights from local flattening

The local flat filter normally excludes very dark shadows and very bright highlights because aggressive flattening in those regions can suppress legitimate texture or amplify noise. To narrow the eligible tonal range further, for example:

```powershell
--flat-highpass 303030 `
--flat-lowpass e8e8e8
```

PowerShell examples use hashless hexadecimal values. The thresholds are converted to equivalent perceptual Lab lightness; they are not simple per-channel RGB clipping thresholds.

### 12.6 Allow cleanup across the full tonal range

For a controlled test where you intentionally want the local/profile stages to operate from black to white:

```powershell
--flat-highpass 000000 `
--flat-lowpass ffffff
```

This is more aggressive than the defaults. In very dark areas it can expose noise or low-signal chroma errors, so compare carefully against the protected default range.

### 12.7 Visible vertical bands

Use this when the artifact is predominantly vertical in the displayed image rather than horizontal:

```powershell
--band-axis vertical
```

Flicker Suppressor rotates the image internally into the row-oriented domain, runs the normal restoration, then rotates the result back. Auto mode normally handles ordinary portrait/landscape orientation, but manual override is useful for cropped, externally rotated, or ambiguous files.

### 12.8 Residual bands in both horizontal and vertical directions

Use this when one orientation has been corrected well but a second orthogonal stripe pattern is still visible:

```powershell
--orthogonal-profile `
--orthogonal-profile-luma-strength 0.60 `
--orthogonal-profile-chroma-strength 0.50
```

Orthogonal cleanup is independently opt-in and reuses the robust residual-profile algorithm in the perpendicular direction; it does not run the neural models twice. The main `--flat-profile` switch does not need to be enabled. `--band-axis both` remains accepted as a legacy shortcut for horizontal primary processing plus this orthogonal pass.

### 12.9 Very broad/few-cycle residuals on one dominant surface

Use this only when a large surface such as a wall still shows a few extremely broad bands or waves that ordinary period-based cleanup cannot distinguish reliably from real illumination gradients.

```powershell
--flat-surface-equalizer
```

The large-surface equalizer makes a stronger assumption than the normal profile filter: it identifies one dominant, color-consistent surface and fits a low-order illumination/color baseline through it. Inspect the result carefully, and use `--debug-dir` if you want to verify the selected surface mask.

### 12.10 If the local flat filter alters people, objects, or small smooth regions

The current local filter includes extent, same-surface color, fill-ratio, and edge-distance safety gates. If a small smooth object is still visibly affected, first make the edge protection wider:

```powershell
--flat-local-edge-distance 10
```

If necessary, make same-surface membership stricter by lowering the local color tolerances. Use `--debug-dir` to inspect `*_flat_local_safe_gate.png`, `*_flat_local_color_gate.png`, `*_flat_local_fill_gate.png`, and `*_flat_local_edge_distance_gate.png` before making large threshold changes.

### 12.11 Localize a cleanup stage with a GUI paint mask

Use a stage mask when a cleanup algorithm is useful in one part of the frame but unnecessarily changes another part. This is especially useful for aggressive Residual profile settings that are needed on a banded foreground subject but should not act across the entire background.

1. Enable the target cleanup stage.
2. Press **Mask** directly below that stage's On/Off switch.
3. Paint the regions where that stage is allowed to appear. The button reads **Settings** while mask mode is active; press it to return to the normal controls.
4. Use Brush/Eraser Size, Feather, and Opacity to shape the gate. Feather is an absolute source-pixel width outside the solid core.
5. Preview again. The automatic stage still estimates its correction from the image normally, but only the painted alpha is allowed into the final result.

The first time a cleanup stage/mask is enabled, its mask is authored at 100% alpha over the whole image. This is initially equivalent to unrestricted processing, and Eraser is selected by default so the user can immediately remove the regions that should be protected. If all alpha is erased, the authored empty mask disables that stage everywhere until Brush or Invert adds coverage again. Masks are GUI session data rather than CLI arguments and are not included in developer JSON settings export.

### 12.12 Choosing an aggressiveness level

A practical progression is:

```text
1. Normal restoration
   -> default one-pass neural correction

2. Strong restoration
   -> --passes 2 --flat-filter

3. Residual periodic cleanup
   -> add --flat-profile with strengths around 0.50

4. Extreme fine periodic cleanup
   -> raise profile strengths gradually and optionally force a measured period

5. Special geometry
   -> --band-axis vertical / both, or --flat-surface-equalizer for very broad dominant-surface residuals
```

Avoid enabling every stage at maximum strength by default. The safest result is normally the least aggressive configuration that removes the visible artifact.

---

## 13. Full inference option reference

The defaults below are taken directly from the current parser. For PowerShell color values, examples use hashless hexadecimal notation.

### 13.1 Core input/output and runtime

| Option | Default | Explanation | Example / tuning note |
|---|---:|---|---|
| `-h/--help` | - | show this help message and exit | Run `python .\hybrid_infer_detail_preserving.py --help`. |
| `--input` | required | Input image file or directory. Directory input is scanned recursively for supported raster formats. | Example: `--input .\photo.jpg` or `--input .\photos`. |
| `--output` | required | Output PNG path for a single image, or output directory for directory input. | Example: `--output .\photo_restored.png` or `--output .\restored`. |
| `--luma-model` | required | Original/standard 3-channel Restormer final.pth | Release model: `--luma-model .\models\y.pth`. |
| `--chroma-model` | required | Fine-tuned 2-channel CbCr branch | Release model: `--chroma-model .\models\chroma.pth`. |
| `--device` | `auto` | PyTorch device. `auto` chooses CUDA device 0 first, then Apple MPS, then CPU. Explicit CUDA indices such as `cuda:0` and `cuda:1` are supported. | Use `--device cuda:1` to choose the second visible NVIDIA GPU. The GUI enumerates detected CUDA GPUs by model name and labels Auto/CPU hardware. |
| `--amp` / `--no-amp` | on | Enable/disable CUDA FP16 autocast. AMP is used only when the resolved device is CUDA; on CPU/MPS the flag has no effect. | Default is on for CUDA. Use `--no-amp` for an FP32 comparison. |
| `--processing-size` | `512` | Square neural-network working resolution. Must be a multiple of 8 in the range 256–2048. Corrections are resized back to the source resolution. | Keep `512` unless benchmarking a deliberate change. |
| `--overwrite` | off | Overwrite existing output files. Without it, existing outputs are skipped. | Add `--overwrite` when rerunning a batch into the same output directory. |
| `--debug-dir` | none | Directory for correction maps, masks, gates, and period-analysis diagnostics. | Example: `--debug-dir .\debug_photo`. |

### 13.2 Band orientation and orthogonal cleanup

| Option | Default | Explanation | Example / tuning note |
|---|---:|---|---|
| `--band-axis` | `auto` | Visible primary band direction. Choices: `auto`, `horizontal`, `vertical`, `both`. `both` is retained as a legacy shortcut for horizontal + orthogonal cleanup. | Prefer `auto`, `horizontal`, or `vertical`; use `--orthogonal-profile` for mixed-axis residuals. |
| `--band-axis-auto-aspect-ratio` | `1.1` | Portrait/landscape H/W ratio used by --band-axis auto; near-square images use a coarse striping score | Raise above 1.10 if Auto should classify fewer mildly portrait images as vertical. |
| `--band-axis-analysis-size` | `384` | Maximum side used only for the near-square auto-axis diagnostic | Usually leave at 384; only affects near-square Auto diagnosis. |
| `--orthogonal-profile` | off | Enable a robust residual-profile cleanup in the axis perpendicular to the primary band direction. | Use for mixed-axis residuals; independent of `--flat-profile` and `--flat-filter`. |
| `--orthogonal-profile-luma-strength` | `-1.0` | Orthogonal-profile Y strength; negative = reuse --flat-profile-luma-strength | Example: `--orthogonal-profile-luma-strength 0.70`. |
| `--orthogonal-profile-chroma-strength` | `-1.0` | Orthogonal-profile CbCr strength; negative = reuse --flat-profile-chroma-strength | Example: `--orthogonal-profile-chroma-strength 0.60`. |
| `--orthogonal-profile-band-period` | `0.0` | Orthogonal-profile period override in perpendicular-axis pixels; 0 = auto | Force a known perpendicular-axis period, e.g. `--orthogonal-profile-band-period 220`. |

### 13.3 Pass count, exposure and neural correction

| Option | Default | Explanation | Example / tuning note |
|---|---:|---|---|
| `--passes` | `1` | Use 2 only for unusually severe residual bands Choices: `1`, `2`. | Use `--passes 2` only when one pass leaves strong residual bands. |
| `--second-pass-strength` | `1.0` | Scale both Y and chroma corrections on pass 2 | Example: `0.70` for a gentler second pass. |
| `--exposure-lock` | `pass2` | CLI control for removing global DC from the Y correction. Choices: `off`, `pass2`, `all`. | CLI default is `pass2`. The desktop GUI hides this option and forces `all`. |
| `--luma-mode` | `directional` | How the Restormer Y prediction is converted into a correction field: raw, directional, directional-additive, or row. Choices: `raw`, `directional`, `directional-additive`, `row`. | Recommended: `directional`. Use `raw` only for legacy comparison. |
| `--horizontal-sigma` | `16.0` | Horizontal Gaussian sigma used to constrain the neural luminance correction field at processing resolution. | Larger values preserve more original horizontal texture in the applied correction; 16 is the tested default. |
| `--vertical-sigma` | `0.0` | Optional vertical smoothing sigma for the neural Y correction field. Zero preserves the estimated row-frequency structure. | Normally 0. Increasing it can suppress very fine correction variation but may under-correct fine bands. |
| `--luma-eps` | `0.02` | Positive stabilization offset used by log-gain luminance correction, especially in deep shadows. | Normally leave 0.02. Increasing stabilizes very dark pixels but changes gain behavior. |
| `--clip-stops` | `2.0` | Maximum absolute log-gain correction in exposure stops before application. | Lower for conservative correction; default allows up to ±2 stops in the model correction field. |
| `--no-row-anchor` | off | Disable preservation of each row's mean neural correction after horizontal filtering. | Mostly diagnostic; the default anchored behavior is recommended. |
| `--luma-strength` | `1.0` | Global multiplier for the neural luminance correction. | Example: `--luma-strength 0.8` for a globally gentler neural Y correction. |
| `--chroma-strength` | `1.0` | Global multiplier for the neural Cb/Cr correction. | Example: `--chroma-strength 0.8` if chroma correction is globally too aggressive. |
| `--highlight-recovery-strength` | `1.0` | Fraction of original near-white luminance restored when processing made it darker. | GUI **Highlight recovery**. `0` disables; `1` restores the full lost source Y inside the highlight gate. |
| `--highlight-recovery-start` | `0.90` | Original/source Y where highlight recovery begins to fade in. | Advanced tuning only. Raise to restrict recovery to more extreme highlights. |
| `--highlight-recovery-full` | `0.99` | Original/source Y where the requested highlight-recovery strength reaches full weight. | Must be greater than `--highlight-recovery-start`; normally leave at `0.99`. |

### 13.4 Local flat-region filter

| Option | Default | Explanation | Example / tuning note |
|---|---:|---|---|
| `--flat-filter` | off | Enable optional flat-region residual band post-filter | Add this for residual bands on broad low-detail surfaces. |
| `--flat-band-period` | `0.0` | Band period in full-resolution pixels; 0 = estimate from model corrections | Force the local filter period, e.g. `--flat-band-period 100`; 0 uses model-correction estimation. |
| `--flat-period-sigma-ratio` | `0.25` | Vertical smoothing sigma as a fraction of estimated band period | Controls vertical smoothing scale relative to period. Higher = broader flattening. |
| `--flat-horizontal-sigma` | `16.0` | Horizontal coherence support in full-resolution pixels | Higher values require broader horizontal coherence before structure is treated as band-like. |
| `--flat-full` | `0.007` | Detail metric below this gets full flat-region weight | Lowering makes full local-flat confidence harder to reach; advanced tuning only. |
| `--flat-none` | `0.025` | Detail metric above this gets zero flat-region weight | Lowering rejects textured regions sooner; advanced tuning only. |
| `--flat-luma-strength` | `0.7` | Strength of the local flat-region Y correction. | Default 0.70. Raise cautiously if the local wall correction is too weak. |
| `--flat-chroma-strength` | `0.85` | Strength of the local flat-region Cb/Cr correction. | Default 0.85. Raise cautiously for residual color bands on safe flat surfaces. |
| `--flat-cleanup-strength` | `1.0` | Legacy CLI-only multiplier applied to both local strengths. | Kept at 1.0 by the GUI; use the direct luminance/chroma controls instead. |
| `--flat-highpass` | `#232323` | Dark-side Lab-lightness cutoff; darker tones are excluded. Use quoted #RRGGBB or 'off'. | PowerShell example: `--flat-highpass 232323`; use `000000` for effectively no dark exclusion or `off` to disable. |
| `--flat-lowpass` | `#efefef` | Bright-side Lab-lightness cutoff; brighter tones are excluded. Use quoted #RRGGBB or 'off'. | PowerShell example: `--flat-lowpass efefef`; use `ffffff` for effectively no bright exclusion or `off` to disable. |

The GUI exposes these as Shadow cutoff and Highlight cutoff. On edit completion it canonicalizes valid six-digit RGB text to `#RRGGBB`; invalid Shadow input falls back to `#000000`, invalid Highlight input falls back to `#FFFFFF`, and inconsistent ordering is corrected to the corresponding safe endpoint. The GUI does not expose the CLI-only `off` text sentinel.
| `--flat-shadow-ramp` | `12.0` | Smooth shadow transition width around --flat-highpass in CIE Lab L* units | Larger values make the dark cutoff fade over a wider L* range. |
| `--flat-highlight-ramp` | `8.0` | Smooth highlight transition width around --flat-lowpass in CIE Lab L* units | Larger values make the bright cutoff fade over a wider L* range. |
| `--flat-luma-spatial-feather` | `1.25` | Small spatial feather sigma for the tone gate; 0 disables | Set to 0 to disable the small spatial feather around the tone gate. |
| `--flat-base-lstar-sigma-ratio` | `0.4` | Vertical smoothing scale for band-resistant base lightness as fraction of band period | Higher values estimate base surface lightness over a broader vertical scale. |
| `--flat-edge-low` | `0.018` | Scene-edge barrier starts at this smoothed Y/C gradient | Lower values make edge protection start sooner. |
| `--flat-edge-high` | `0.055` | Scene-edge barrier is fully closed at this smoothed Y/C gradient | Lower values fully block smoothing at weaker edges. |
| `--flat-edge-guard` | `2` | Pixels of support-only guard around strong scene edges | Increase if foreground edges still bleed into nearby wall corrections. |
| `--flat-broad-structure-sigma` | `8.0` | Broad scale used to detect soft/defocused scene boundaries for local flat cleanup | Increase only for extremely broad blur; `0` disables this guard for regression testing. |
| `--flat-broad-structure-low` | `0.025` | Scale-normalized broad gradient where soft-structure protection begins | Lower values protect weaker blurred boundaries. |
| `--flat-broad-structure-high` | `0.080` | Scale-normalized broad gradient where soft-structure protection is full | Lower values fully block local flattening at weaker blurred boundaries. |
| `--flat-broad-structure-guard` | `8` | Extra pixels protected around detected broad/defocused structure | Increase to widen the protected zone. |
| `--flat-broad-structure-feather` | `6.0` | Feather sigma of the broad/defocus structure guard | Increase for a softer transition. |
| `--flat-coarse-preblur` | `1.0` | Preblur used only to segment textured close-colored surfaces | Increase to make rough texture look more like one underlying surface for segmentation. |
| `--flat-coarse-blend` | `0.25` | How much coarse surface segmentation contributes to the visible local blend | Raise cautiously to let textured surfaces receive more local flattening. |
| `--flat-blend-feather` | `0.75` | Spatial feather sigma for the final correction blend mask | Higher values soften local blend boundaries more broadly. |

### 13.5 Local flat-filter safety gates

| Option | Default | Explanation | Example / tuning note |
|---|---:|---|---|
| `--flat-local-extent-sigma` | `32.0` | Large-neighborhood scale used to reject tiny isolated local-flat islands such as smooth skin patches | Larger neighborhood = stronger rejection of small smooth islands. |
| `--flat-local-extent-low` | `0.18` | Large-surface occupancy where local flattening begins to fade in | Advanced occupancy threshold; lower values allow local correction to start sooner. |
| `--flat-local-extent-high` | `0.5` | Large-surface occupancy where local flattening is fully enabled | Advanced occupancy threshold; lower values reach full local correction sooner. |
| `--flat-local-color-sigma` | `44.0` | Broad neighborhood scale for same-surface color consistency in the local flattener | Larger scale demands broad color consistency over a wider area. |
| `--flat-local-color-preblur` | `2.0` | Preblur before local same-surface color comparison | Pre-smoothing before broad color comparison; larger values ignore more fine texture. |
| `--flat-local-color-luma-tolerance` | `7.0` | Allowed broad-surface Lab L* deviation for local flattening | Lower values require stricter same-surface Lab lightness consistency. |
| `--flat-local-color-chroma-tolerance` | `0.03` | Allowed broad-surface CbCr distance for local flattening | Lower values require stricter same-surface color consistency. |
| `--flat-local-fill-sigma` | `36.0` | Neighborhood scale for same-surface fill ratio | Larger values measure same-surface occupancy over a broader window. |
| `--flat-local-fill-low` | `0.38` | Same-surface fill fraction where local flattening begins to fade in | Advanced fill threshold where correction begins to fade in. |
| `--flat-local-fill-high` | `0.68` | Same-surface fill fraction where local flattening is fully enabled | Advanced fill threshold where correction becomes fully eligible. |
| `--flat-local-edge-distance` | `6` | Local-filter-only guard distance from strong scene/object edges; valid range 0–100. `0` disables this guard. | Increase to protect a wider zone around objects/people from the local filter. Negative values have no separate meaning. |
| `--flat-local-edge-feather` | `3.0` | Soft feather applied to the local edge-distance safety gate | Increase to make the edge-distance protection transition softer. |
| `--flat-local-correction-horizontal-sigma` | `128.0` | Processing-X regularization sigma applied to the estimated local correction field for resolved horizontal displayed bands | Reduces long silhouette trails caused by holes in local support. Automatically bypassed for resolved vertical displayed bands to avoid light/fixture streaks; set `0` for unregularized comparison. |
| `--flat-local-application-horizontal-sigma` | `48.0` | Horizontal regularization sigma applied to the final blended local correction field | Prevents protected object masks from becoming hard holes/halos in a row-varying correction. Set `0` for v0.39 application behavior. |

### 13.6 Global residual profile

| Option | Default | Explanation | Example / tuning note |
|---|---:|---|---|
| `--flat-profile` | off | Enable optional residual 1-D row-profile suppression for faint globally coherent bands | Use for residual coherent bands even on textured surfaces. |
| `--flat-profile-luma-strength` | `0.35` | Residual row-profile suppression strength for log-Y when --flat-profile is enabled | 0.50 is a strong practical preset; 0.8-1.2 may help extreme fine bands. |
| `--flat-profile-chroma-strength` | `0.35` | Residual row-profile suppression strength for CbCr when --flat-profile is enabled | 0.50 is a strong practical preset; increase less aggressively than Y when possible. |
| `--flat-profile-narrow-ratio` | `0.035` | Noise-suppression scale of residual profile as fraction of band period | Controls noise suppression on the 1-D residual profile. Advanced tuning only. |
| `--flat-profile-base-ratio` | `0.4` | Baseline scale of residual profile as fraction of band period | Controls how much slow variation is treated as legitimate baseline rather than flicker. |
| `--flat-profile-band-period` | `0.0` | Override only the residual-profile period in full-resolution pixels; 0 = multiscale auto; manual values are limited to 1–7680 px. | Force a known period such as `37`; 0 uses harmonic-aware auto detection. |
| `--flat-profile-period-mode` | `multiscale` | Period estimator for --flat-profile; multiscale is texture tolerant and harmonic aware Choices: `multiscale`, `legacy`. | Keep `multiscale` normally. `legacy` is retained for comparison/regression. |
| `--flat-profile-period-min` | `12.0` | Smallest period considered by multiscale profile detection | Lower only if you expect extremely fine bands below 12 px. |
| `--flat-profile-period-max-fraction` | `0.6` | Largest profile period as a fraction of image height | Raise if legitimate band periods span more than 60% of the image height. |
| `--flat-profile-period-analysis` | `512` | Internal row-spectrum analysis height | Internal spectral-analysis height. Usually leave at 512. |
| `--flat-profile-huber-k` | `2.5` | Robust row-consensus outlier scale; lower rejects objects/texture more aggressively | Lower = stronger rejection of objects/outliers in row consensus; too low can reduce usable support. |
| `--flat-profile-min-coverage` | `0.08` | Minimum trustworthy horizontal support fraction for a row profile | Raise to require more trustworthy horizontal evidence before applying a row correction. |
| `--flat-profile-adaptive` / `--no-flat-profile-adaptive` | on | Fit a slowly varying local multiplier for the globally estimated profile waveform | Keep on normally; disable only to reproduce the legacy single-amplitude profile. |
| `--flat-profile-adaptive-x-ratio` | `0.35` | Horizontal smoothing scale of the local amplitude fit as a fraction of band period | Larger values make profile strength vary more slowly across the image. |
| `--flat-profile-adaptive-y-ratio` | `1.50` | Vertical fitting/smoothing scale of the local amplitude fit as a fraction of band period | Larger values make the local gain less sensitive to short vertical regions. |
| `--flat-profile-adaptive-corr-low` | `0.15` | Band-limited local waveform correlation where adaptive evidence begins to fade in | Raise to require stronger local agreement before profile correction appears. |
| `--flat-profile-adaptive-corr-high` | `0.45` | Band-limited local waveform correlation where adaptive evidence is fully trusted | Keep above the low threshold; lower values make the adaptive profile more permissive. |
| `--flat-profile-adaptive-max-gain` | `1.00` | Maximum local multiplier of the globally estimated profile waveform | Default 1.0 is attenuation-only: local adaptation can reduce the requested profile but cannot exceed its selected strength. |
| `--flat-profile-no-harm` / `--no-flat-profile-no-harm` | on | Validate the actual local profile correction and suppress it where band-limited residual energy would not clearly decrease. | Keep on normally. Disable only for regression comparison with the earlier adaptive-profile behavior. |

### 13.7 Dominant large-surface equalizer

| Option | Default | Explanation | Example / tuning note |
|---|---:|---|---|
| `--flat-surface-equalizer` | off | Enable v1.7 dominant large-surface row equalizer for extremely broad/few-cycle residual bands | Use only for very broad/few-cycle residuals on one dominant surface. |
| `--flat-surface-equalizer-luma-strength` | `1.0` | Large-surface equalizer strength for log-luminance | Reduce below 1.0 if the equalizer over-flattens real illumination. |
| `--flat-surface-equalizer-chroma-strength` | `1.0` | Large-surface equalizer strength for CbCr | Reduce below 1.0 if broad color correction is too aggressive. |
| `--flat-surface-equalizer-degree` | `2` | Polynomial degree of the legitimate large-surface illumination/color baseline (default quadratic) | Keep 2 normally. Higher degrees can start fitting the flicker itself. |
| `--flat-surface-equalizer-preblur` | `4.0` | Preblur used to recognize a rough textured surface as one underlying surface | Increase to recognize rougher texture as one underlying surface. |
| `--flat-surface-equalizer-analysis` | `256` | Maximum side of low-resolution dominant-surface connected-component analysis | Larger = finer connected-component analysis but more compute. |
| `--flat-surface-equalizer-threshold` | `0.3` | Low-resolution candidate threshold for the dominant surface | Higher makes candidate-surface selection stricter. |
| `--flat-surface-equalizer-close-radius` | `1` | Small low-resolution morphological closing radius for holes in textured surfaces | Increase carefully to bridge small holes in the candidate surface. |
| `--flat-surface-equalizer-min-area` | `0.08` | Minimum image-area fraction required before a surface can be equalized | Raise to ensure only very dominant surfaces are considered. |
| `--flat-surface-equalizer-chroma-tolerance` | `0.08` | CbCr distance tolerance used to keep the dominant large surface color-consistent | Lower to keep the selected dominant surface more color-consistent. |
| `--flat-surface-equalizer-luma-tolerance` | `0.24` | Coarse Y tolerance used to separate the dominant surface from different objects | Lower to separate surfaces with different coarse brightness more strictly. |
| `--flat-surface-equalizer-row-edge-barrier` | `0.03` | Row-spanning scene-edge density that splits otherwise reconnecting large surfaces | Lower to split surfaces at weaker row-spanning boundaries. |
| `--flat-surface-equalizer-row-edge-guard` | `3` | Vertical pixels guarded around a row-spanning surface boundary | Increase to guard more pixels around detected row-spanning boundaries. |
| `--flat-surface-equalizer-feather` | `5.0` | Full-resolution feather sigma of the selected surface boundary | Higher values soften the selected-surface boundary more broadly. |
| `--flat-surface-equalizer-row-sigma` | `2.0` | Small 1-D smoothing of robust per-row surface measurements before baseline fitting | Higher values smooth per-row surface measurements more before polynomial fitting. |
| `--flat-surface-equalizer-huber-k` | `2.5` | Robust cross-column outlier scale for the large-surface row estimate | Lower rejects cross-column outliers more aggressively in the large-surface estimate. |
| `--flat-surface-equalizer-min-coverage` | `0.04` | Minimum horizontal coverage of the selected surface required on a row | Raise to require more selected-surface coverage before trusting a row. |
| `--flat-allow-mean-shift` | off | Allow the local flat filter to change global Y/CbCr means | Normally leave off so local filtering preserves global Y/CbCr means. |


---

## 14. Reading the console diagnostics

Typical lines include:

```text
device: cuda; AMP: True; passes=2; exposure-lock=pass2; band-axis=auto
axis: horizontal (...); scores H=... V=...
pass 1: corr-rms=... exposure-removed=... stops gamut-compressed=...%
pass 2: corr-rms=... exposure-removed=... stops gamut-compressed=...%
flat-filter: blend>0.5=... support>0.5=... local-period=... profile-period=... conf=...
profile period candidates: 100px:0.83, 50px:0.61, ...
highlight-recovery: strength=1.00 gate=...% mean-Y-restored=... gamut-compressed=...%
```

Interpretation:

- `corr-rms` - magnitude of the constrained neural Y correction;
- `exposure-removed` - global Y DC removed by exposure locking;
- `gamut-compressed` - percentage of pixels whose chroma was reduced to preserve requested Y in RGB gamut;
- `blend>0.5` - fraction receiving strong local flat-filter correction;
- `support>0.5` - fraction allowed to influence local correction estimation;
- `local-period` - period used by the local flat filter;
- `profile-period` - period used by the robust row profile;
- `conf` - profile-period confidence;
- `profile-support` - trustworthy row-profile evidence coverage;
- `local-safe` - fraction passing local same-surface safety;
- `surface-eq` - fraction strongly affected by the large-surface equalizer.

---

## 15. Troubleshooting

### CUDA requested but unavailable

Run:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If `False`, verify that you installed a CUDA-enabled PyTorch wheel and have a compatible NVIDIA driver. Installing the full standalone CUDA Toolkit is not required for the prebuilt wheel.

### PowerShell breaks at a hex color

Do not write an unescaped `#` in an unquoted PowerShell argument. The safest convention in this project is hashless hex:

```powershell
--flat-highpass 232323 `
--flat-lowpass efefef
```

### Result becomes globally brighter on two passes

Keep the default:

```powershell
--exposure-lock pass2
```

If desired, also reduce:

```powershell
--second-pass-strength 0.70
```

### Fine details are altered by the local flat filter

The current safety gates, including the broad/defocus structure guard, are designed to prevent this. If an edge/object is still affected, first try increasing:

```powershell
--flat-local-edge-distance 10
```

or making same-surface membership stricter by lowering the color tolerances. Use `--debug-dir` to inspect `flat_local_safe_gate` and related masks.

### Strong Residual profile creates halos or bright blobs near a background boundary

The current high-strength path uses a band-coherent no-harm gate, dominant-surface handoff, one-sided boundary transition, bright soft-edge cap, and highlight-headroom limiter. If an unusual scene still produces a local artifact, first compare against a lower profile strength. If the high strength is genuinely needed on a subject, use the **Residual profile Mask** in the GUI to paint only the subject/region that needs the extra correction; this is safer than weakening the automatic background protections globally.

### Cleanup should affect only one part of the image

Use the corresponding GUI **Mask** for Flat-region cleanup, Residual profile, or Broad residual cleanup. The mask is a final application gate, so it restricts the visible result without forcing the period/surface estimator to work from a small painted crop.

### Residual bands remain but the surface is textured

Use `--flat-profile`; do not simply increase local flat-filter strength.

### Very fine regular bands remain

Raise profile strength and optionally force the measured period:

```powershell
--flat-profile `
--flat-profile-band-period 37 `
--flat-profile-luma-strength 1.00 `
--flat-profile-chroma-strength 0.80
```

### Bands are vertical

Use `--band-axis vertical`. If both orientations are present, keep the appropriate primary direction and add `--orthogonal-profile`.

### Broad/few-cycle residual remains on one wall

Try `--flat-surface-equalizer`; inspect the selected region in the debug output before making it part of a default workflow.

---

## 16. Supported image I/O

Input extensions:

```text
.png .jpg .jpeg .bmp .tif .tiff .webp
```

Input EXIF orientation is applied before processing. Output is currently 8-bit RGB PNG. Directory input is recursive and preserves the relative directory structure under the output directory.

---

## 17. Model and training lineage

The release models were developed from the BurstDeflicker/Restormer lineage.

- The standard single-image RGB Restormer was adapted/fine-tuned for single-image flicker restoration using BurstFlicker data and is used in this release only as the luminance-correction estimator.
- The chroma branch was developed from the chroma-refined Restormer lineage, structurally converted to a two-channel Cb/Cr output head, and fine-tuned to separate chromatic flicker from luminance-only bands and neutral/grayscale cases.
- The deterministic inference stages were added after empirical testing exposed detail loss, recursive exposure drift, chroma hallucination, edge bleed, textured-surface residuals, orientation changes, small smooth-object false positives, defocused-boundary artifacts, and high-strength profile halos on smooth backgrounds.

---

## 18. License and attribution

Flicker Suppressor is released under the Apache License 2.0.

See:

- `LICENSE` - top-level Apache License 2.0;
- `LICENSE-THIRD-PARTY` - development/source-tree notices covering Restormer, BurstDeflicker/BasicSR, data/model lineage, and development dependencies;
- `RELEASE-LICENSE-THIRD-PARTY` - release-specific notices and license texts for the third-party runtimes and native libraries bundled in the frozen Windows portable build.

Important upstream projects:

- Restormer: https://github.com/swz30/Restormer - MIT License.
- BurstDeflicker: https://github.com/qulishen/BurstDeflicker - Apache License 2.0.
- BasicSR: https://github.com/XPixelGroup/BasicSR - Apache License 2.0.

---

## 19. Desktop GUI reference

The desktop application is implemented with PySide6/Qt 6 and uses the same in-process inference engine and model checkpoints as the CLI. The GUI deliberately exposes a curated subset of the parser while filling every hidden processing parameter from the authoritative CLI defaults.

Startup is intentionally split so visual feedback appears before the heavy application import chain. `gui_main.py` creates the Qt application and a small frameless/translucent splash first, using `assets/logo_256.png` without importing the normal GUI package. The centered logo is followed by a separate bold white **`loading...`** label below it; the label uses a blurred dark drop shadow so it remains readable against bright desktop/window content behind the translucent splash. Its typography is scoped specifically to the splash label so the later application-wide dark stylesheet cannot resize it during the final startup frames. Only after the splash has been shown and Qt events have been processed are `MainWindow` and the application theme imported. There is no artificial minimum splash duration; it closes immediately after the main window is constructed and shown.

### 19.1 Current image vs. selected images

The bottom image strip supports extended multi-selection. These concepts are intentionally separate:

- **selected images** are the batch selection used by actions such as Close selected, Export selected, and Paste editing settings to selected;
- the **current image** is the one shown on the canvas and represented by the controls in the right panel.

The current thumbnail has a **red border**. Blue item styling continues to indicate the multi-selection. Processing state is also reflected directly in the strip: while a document is active in Preview / Apply or is the current export job, its thumbnail image receives a neutral grey translucent veil and an animated 12-spoke spinner. This is a scaled-down counterpart of the canvas processing feedback; the red current-image border is painted above it so current-image identity remains visible.

Changing a setting in the right panel changes **only the current image**. It does not batch-apply that setting merely because multiple thumbnails are selected. Copy requires one selected activated source image; Paste applies the copied recipe to all selected targets and activates them, but does not copy painted masks.

When a new image batch is imported, the previous selection is cleared, only the newly imported images are selected, and the **first newly imported image** becomes current without collapsing the new multi-selection.

Supported GUI imports are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, and `.webp`. Files can be opened from the File menu / Open recent or dropped onto the main canvas or bottom strip.

### 19.2 Per-image activation

Each imported image has a per-image checkbox directly below the Editing settings heading:

**Activate Flicker Suppressor for this image**

It is off by default. While off:

- all editing controls are disabled/greyed out;
- Preview / Apply is disabled;
- the image is excluded from Export selected and Export all;
- comparison controls requiring an edited result are unavailable.

Activation does not delete the image's stored settings or an existing cached preview. Re-enabling an unchanged image can reuse that cache.

This activation flag, rather than the existence of a preview, defines whether an image is considered part of the restoration/export set.

### 19.3 Visible settings

The current GUI groups controls as follows.

#### Restormer correction

1. Band direction - `Auto`, `Horizontal`, `Vertical`;
2. Device - dynamic hardware-labelled Auto/CUDA/CPU entries and MPS when available;
3. Use FP16 / AMP - checked by default, editable for Auto/CUDA, greyed out for explicit CPU/MPS;
4. Processing size - default `512`, constrained to `256-2048` and normalized to a multiple of 8;
5. Passes - `1` or `2`;
6. Second-pass strength - enabled only for two passes;
7. Luminance mode - `Directional`, `Directional additive`, `Row`, `Raw`;
8. Highlight recovery - default `1.0`, range `0-1`.

The GUI hides `--exposure-lock` and forces `all`. The highlight gate endpoints remain advanced CLI-only settings.

#### Flat-region cleanup

- Enable flat-region cleanup;
- Flat luminance strength - default `0.70`, GUI range `0-2`;
- Flat chroma strength - default `0.85`, GUI range `0-2`;
- Shadow cutoff;
- Highlight cutoff;
- Object-edge protection distance.

Object-edge protection distance is local-filter-only and is disabled when Flat-region cleanup is off. Valid GUI/CLI range is `0-100`; `0` disables that guard.

#### Residual profile

- Enable residual profile;
- Profile luminance strength;
- Profile chroma strength;
- Profile band period (px), with an **Auto** checkbox;
- Enable orthogonal cleanup;
- Orthogonal luminance strength;
- Orthogonal chroma strength.

Main Residual profile is independent of Flat-region cleanup. Orthogonal cleanup reuses the same residual-profile algorithm in the perpendicular axis but is independently opt-in and does not require the main Residual profile checkbox.

When Profile band period Auto is checked, the effective value passed to inference is `0`, which means harmonic-aware automatic detection. When Auto is unchecked, manual periods are limited to `1-7680` full-resolution pixels.

#### Broad residual cleanup

- Enable large-surface equalizer.

This stage is independent of the local Flat-region cleanup and the main Residual profile switch.

#### Per-stage paint masks

The **Flat-region cleanup**, **Residual profile**, and **Broad residual cleanup** group boxes each have a **Mask** button directly under the stage On/Off switch. The On/Off switch remains visible while painting. Entering mask mode swaps the remainder of that section to Brush/Eraser controls. While the editor is active the button label changes from **Mask** to **Settings**; pressing **Settings** exits mask mode and restores the section's normal controls.

Each stage owns its own Brush/Eraser state. Brush and Eraser independently remember:

- **Size**: `1-1000 px`, default `160 px`; this is the diameter of the solid core;
- **Feather**: `0-1000 px`, default `32 px`; this is an absolute source-image-pixel falloff outside the core and is independent of Size;
- **Opacity**: `1-100%`, default `100%`.

The tool row contains **Brush**, **Eraser**, and **Invert**. Eraser is selected by default for a newly opened mask because new masks begin at full 100% coverage. Brush and Eraser labels include dedicated vector SVG icons from the application `assets` directory so the tool shape is easy to recognize and remains sharp under high-DPI scaling. The setting-reset icon and the numeric spin up/down chevrons are SVG assets for the same reason. Invert is an immediate mask operation rather than a persistent drawing tool: every mask alpha value is replaced with `1 - alpha` (equivalently `255 - alpha` in the stored 8-bit mask). Inverting a freshly created full-coverage mask therefore makes it empty; inverting it again restores full coverage. Invert does not change Brush/Eraser Size, Feather, or Opacity.

For example, a 20 px brush with 100 px Feather and a 200 px brush with 100 px Feather both have exactly a 100 px feather zone. The canvas cursor shows the solid-core boundary and, when Feather is non-zero, a second dashed outer feather boundary. The stroke itself is rasterized as a continuous capsule along the mouse path, avoiding gaps even when Feather is much larger than Size.

Brush opacity is **non-building**. It is treated as a target alpha, not an amount of paint to add: 50% over an existing 20% mask becomes 50%; 20% over an existing 50% mask remains 50%. Repeated overlap, including self-overlap during a drag, cannot push the mask above the selected brush opacity unless a stronger brush is used later.

Eraser uses its own independent Size/Feather/Opacity values. Its strength is evaluated against a snapshot of the mask taken at mouse-down. Therefore crossing the same pixel repeatedly during one mouse-down -> drag -> mouse-up removes the selected amount only once rather than compounding on every sample. Releasing and beginning a new eraser stroke can remove more.

Directly below the Opacity row are **Copy mask** and **Paste mask**. Copy mask stores the current mask pixels together with whether the source mask has been authored. Paste mask replaces the target section's mask with that copied mask, making it possible to reuse exactly the same painted gate across Flat-region cleanup, Residual profile, and Broad residual cleanup. Only mask geometry/alpha and authored state are transferred; Brush/Eraser tool selection and their independent Size/Feather/Opacity settings are not changed. The dedicated mask clipboard is separate from **Copy/Paste editing settings**, which continues to exclude image-specific mask geometry.

Only one cleanup mask can be edited at a time. The canvas automatically switches to **Single** view when mask mode starts and restores the previous Single/Split/Side-by-side state when mask mode ends. The active mask is drawn over the image as a red cast only while mask mode is active; normal viewing never displays the overlay. Display opacity is 70% of the stored mask alpha, so a processing mask at 100% alpha appears as a 70% red overlay, 50% appears as 35%, and 0% remains invisible. This scaling is visualization-only and does not change the mask applied to processing. Left-drag paints or erases, middle-drag continues to pan, and normal zoom behavior is unchanged.

Paint masks are stored per image at full display resolution. The first time a cleanup stage/mask is enabled, Flicker Suppressor lazily authors a full-coverage (`alpha = 1`) mask, which is equivalent to unrestricted legacy behavior but removes the ambiguity between an untouched mask and a deliberately erased one. **Eraser** is the default selected tool. From then on, the stage's final visible correction is multiplied by mask alpha, including soft feather/opacity transitions. Erasing all alpha leaves an authored empty mask and therefore disables that stage everywhere until Brush or Invert adds coverage again.

The masks are final application gates rather than estimation masks. For Flat-region cleanup, the mask is applied after horizontal correction-field regularization so regularization cannot leak visible correction beyond a painted edge; the Flat mask also restricts the dominant-surface ownership handoff used to protect Residual profile. For Residual profile, the same painted mask gates both the main profile correction and the optional Orthogonal cleanup pass. For Broad residual cleanup, the equalizer still fits the full automatically selected large surface, then the painted mask gates only the final equalizer delta.

Masks are session/per-image editing data, not parser options. Editing a mask marks the current preview stale so Preview/Export recomputes the image. Copy/Paste editing settings deliberately does not copy image-specific mask geometry, and the developer JSON export does not embed mask pixels.

#### Export settings (dev)

The **Export json** button writes the complete processing namespace for the current activated image as human-readable UTF-8 JSON.

The JSON includes both visible values and GUI-hidden parser defaults, plus GUI policy such as `exposure_lock = "all"`. Machine-specific plumbing is excluded: input/output paths, model paths, debug directory, and overwrite behavior are not written. Painted cleanup-mask pixels are also not embedded in this JSON because they are image-specific GUI/session data rather than parser settings.

### 19.4 Per-setting reset, editing and validation

Numeric/text settings and slider rows have a reset button on their right. The button restores only that setting to its authoritative default. Slider-backed settings can also be reset by double-clicking the slider.

The larger **Reset** button beside Preview / Apply is an image-level processing reset. It restores the current image's complete processing settings to `default_settings()` and clears that image's cleanup masks/authored-mask state. It does **not** delete an already rendered preview file or remove that preview from the canvas. If a cached preview exists, Reset instead marks it stale (`dirty = true`): the old render stays visible as a reference, but it is no longer considered a valid representation of the current recipe. The next Preview / Apply or Export therefore recomputes from the reset/current settings before treating the result as current. If no cached preview exists, Reset simply leaves the document clean after restoring defaults.

Numeric spin fields use non-live keyboard tracking so locale-style decimal entry such as `0,54` can be completed before the field normalizes/commits. Values are committed when editing finishes.

Mouse-wheel gestures over right-panel numeric inputs, sliders, and dropdowns do **not** modify those controls. The ignored wheel event is left available to the surrounding settings scroll area. Shared combo boxes use a stateful SVG chevron: downward when closed and upward while the popup is open. Clicking the arrow while the popup is already open closes it normally; the click is not replayed into an immediate reopen.

Important GUI validation rules:

- Processing size: clamp to `256-2048`, then normalize to the nearest multiple of 8;
- Object-edge protection distance: clamp to `0-100`;
- manual Profile band period: clamp to `1-7680 px`; Auto uses internal `0`;
- Second-pass strength: `0-2`;
- primary profile luminance/chroma strengths: `0-4`;
- orthogonal profile strengths: `-1-4`, where `-1` means reuse the corresponding primary-profile strength;
- Shadow/Highlight text: canonicalize to `#RRGGBB`, with safe black/white fallback and ordering correction.
- Mask Brush/Eraser Size: `1-1000 px`; Feather: `0-1000 px`; Opacity: `1-100%`; Brush and Eraser keep independent values per cleanup section.

The same normalization is applied before GUI inference and developer JSON export, so older/pasted settings cannot bypass these ranges.

### 19.5 Device enumeration and CUDA selection

The Device dropdown is built from the runtime hardware rather than a static list.

Typical labels are:

```text
auto (NVIDIA RTX 6000)
cuda (NVIDIA RTX 6000)
cuda (NVIDIA RTX 5000)
cpu (Intel Core i9-9980H)
```

Auto resolves to the first visible CUDA GPU when CUDA is available, otherwise MPS when available, otherwise CPU.

Each CUDA entry maps to an exact PyTorch device (`cuda:0`, `cuda:1`, ...). If two visible GPUs have identical model names, the CUDA index is included in the displayed label so they remain distinguishable.

The CLI accepts the same indexed CUDA syntax. An out-of-range CUDA index produces an explicit device error.

### 19.6 Preview cache and comparison modes

**Preview / Apply** processes the current activated image and writes a temporary cached PNG. It does not overwrite the source file. While that job is active, a grey translucent **Processing...** overlay covers the canvas and shows the same spinner/card styling used by the export-progress UI, but without a Cancel button. The overlay is limited to the canvas, blocks canvas interaction for the duration of the preview job, tracks canvas resizing, and is removed on both successful completion and failure. The corresponding thumbnail simultaneously shows a grey translucent processing veil and the same 12-spoke animation at thumbnail scale.

A settings change, painted-mask edit, or image-level Reset can mark the cached render stale. Stale previews are intentionally **kept visible** on the canvas instead of disappearing; this preserves the last render as a comparison/reference while the user adjusts the recipe. Staleness controls reuse, not visibility: Previewing again refreshes the cache, and Export automatically refreshes a stale/missing cache before writing the output.

Once a valid edited preview exists, comparison modes are:

- **Single** - one image, with the eye button toggling Edited/Original;
- **Split** - interactive split comparison;
- **Side by side** - original and edited each own exactly half of the canvas with synchronized zoom/pan.

Canvas zoom controls are **Fit**, **100%**, `-`, `+`; the zoom indicator has fixed width. Mouse wheel zooms the canvas. Middle-mouse drag pans.

### 19.7 Menus and selection actions

Current top-level menus:

- **File** - Open images, Open recent, Close selected, Close all, Export selected, Export all, Exit;
- **Edit** - Copy editing settings from selected, Paste editing settings to selected;
- **Select** - Select all (`Ctrl+A`), Deselect all (`Ctrl+Shift+A`);
- **View** - Zoom to fit, Zoom to 100%, Single view, Split view, Side by side view;
- **Help** - About.

Copy is enabled when exactly one selected image is activated and uses that image as the settings source. Paste applies the copied numeric/boolean editing recipe to **all selected images**, including inactive ones, and activates those targets. Image-specific painted masks are not copied.

### 19.8 Export semantics

The bottom Export button exports the currently selected activated set; the split-button menu exposes Export selected and Export all.

**Export selected...**

```text
selected images ∩ activated images
```

Selected but inactive images are ignored.

**Export all...**

```text
all activated images
```

A prior Preview / Apply is not required. If an activated image has no valid cached render, or its settings changed after the cache was created, it is processed during export and the new result becomes its normal preview cache.

For exactly one exported image, the GUI opens a Save dialog and the user chooses the output PNG filename directly; no `_restored` suffix is forced.

For multiple images, a batch dialog asks for:

- output folder;
- optional filename prefix;
- optional filename suffix.

Before any export processing starts:

1. duplicate target names inside the batch are resolved case-insensitively with ` (1)`, ` (2)`, ` (3)`, ...;
2. every resulting target is checked against the filesystem;
3. each existing target asks **Replace** or **Skip**.

During export, a grey full-window overlay blocks the rest of the application and shows a spinner, **Export in progress...**, and **Cancel**. In parallel, the thumbnail for the image represented by the current export job is marked processing with its grey veil and animated spinner; the marker advances with the queue and is cleared when that job finishes. Cancellation stops the active job and discards the remaining queue. Final copies are committed atomically so a cancelled copy does not leave a partially written destination file.

### 19.9 About / project attribution

The About dialog is text-only and identifies:

- Flicker Suppressor by mattaja;
- Restormer - MIT License;
- BurstDeflicker - Apache License 2.0;
- BasicSR - Apache License 2.0;
- Flicker Suppressor - Apache License 2.0;
- source and releases: https://github.com/GianSegugio/flicker_suppressor.

---

## 20. GUI development and Windows build

### Development launcher

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_gui_dev.ps1
```

The script uses `.venv-gui`, requires Python 3.12, installs/upgrades the tested PyTorch CUDA wheel and `requirements.txt` + `requirements-gui.txt`, then runs `gui_main.py`.

Because the development launcher runs the source tree directly, source/UI changes do not require deleting the `build` directory. Close the running GUI and launch the script again.

### Portable Windows build

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build_windows.ps1
```

The build script:

1. creates/reuses `.venv-gui`;
2. installs the runtime, GUI, and build requirements;
3. creates the Windows icon from the application assets;
4. removes the previous `build` folder;
5. invokes Nuitka standalone mode with the PySide6 plugin and MSVC;
6. bundles PyTorch, NumPy, Pillow, einops, models, GUI assets, README, documentation, and license files.

The portable application folder is produced at:

```text
build\gui_main.dist
```

An end user of the standalone folder does not need a separate Python, PySide6/Qt, PyTorch, or CUDA Toolkit installation. CUDA execution still requires a compatible NVIDIA driver.

For distribution, the portable release should include `LICENSE` and `RELEASE-LICENSE-THIRD-PARTY`. Keep `LICENSE-THIRD-PARTY` with the development/source tree rather than using it as the frozen release notice.

---

*Last Updated: 17 August 2026*
