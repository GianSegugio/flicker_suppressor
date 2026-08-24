# v1.1.0

## Essential editing settings (beta)

- Added **Essential editing settings**, an optional analysis that runs when an image is imported and fills in the settings that must be right before any other tuning matters: band direction, band period, whether the residual profile stage runs, and which profile mode it uses. Everything else is left at its defaults. Enable or disable it from **Edit → Essential editing settings**; it is on by default and the choice is remembered between sessions.
- **This feature is beta and deliberately narrow.** It determines four settings out of roughly 120. It does not choose strengths, pass counts, or safety thresholds, and finding the essential settings is not the same as restoring the image — it points the correction at the right frequency on the right axis, and the rest of the work is still yours. When the evidence is ambiguous the analysis declines rather than guessing.
- Period detection measures on **log-channel ratios** (`log B - log R` and similar) rather than luminance alone. Scene structure is multiplicative and largely cancels in a channel ratio, while a flicker source with a different spectrum from the ambient light does not, so the band stands out far more clearly. Several periods that luminance reports as a harmonic of the true value are resolved correctly this way.
- Every candidate period is cross-checked across multiple search windows and multiple channel ratios. A period that changes when the search window changes is a harmonic-family artifact rather than a measurement, and is reported as undetermined. The agreement rule is deliberately strict: it will decline on images it could have handled rather than risk a confident wrong answer.
- Greyscale images are capped at medium confidence. With no colour ratios available there is no second opinion to check luminance against, and window stability alone cannot detect that luminance is the wrong instrument.
- **Band period** is a dropdown offering **Auto**, each detected candidate, and **Custom…**. Candidates are labelled with their cycle count and marked as harmonics or independent alternatives, because mistaking a second harmonic for the fundamental is the most common way period detection goes wrong. Choosing **Custom…** enables the numeric field. The stored value convention is unchanged: `0` still means Auto.
- While the analysis runs, the window shows a blocking progress overlay matching the export overlay. Dropping further images onto the canvas or the filmstrip still works, so more images can be queued while the first batch is being analysed.
- When the analysis cannot reach a confident answer, that image is reset to defaults and a notice appears the first time it is selected. Resetting to defaults is deliberate: silently carrying the previous image's period onto a new one is worse than starting neutral.
- The command-line equivalent writes the estimate as JSON for later use: `python autosettings.py --input photo.jpg --json photo.settings.json`. Adding `--base defaults.json` merges the estimate into a complete recipe. The exit code is `2` when confidence is too low to use.

## Tone restoration

- Added a **Tone restoration** stage that runs at the end of the pipeline and restores global contrast and gamma lost during processing. The neural passes compress the tonal range — shadows lift and highlights compress — and because this is a global change rather than a band, none of the existing band-focused stages could see or repair it.
- The correction is fitted on band-axis-smoothed envelopes and applied as a smooth additive offset in log space, so the periodic residual passes through unchanged. A tone curve applied directly to a still-banded image would re-expand the bands it was trying to leave alone; with this approach measured band suppression is unchanged to three decimal places.
- Positive corrections are limited by remaining highlight headroom, using the same limiter the residual profile stage already applies. Without it a tone curve drives near-white pixels past the clipping point and turns a graded light source into a flat white disc.
- New GUI section **Tone restoration** with an on/off checkbox (on by default) and **Restoration strength**, default `1.00`. CLI equivalents are `--tone-restore` / `--no-tone-restore` and `--tone-restore-strength`.
- `--tone-restore-max-gain` and `--tone-restore-min-confidence` remain available on the command line but are not shown in the desktop panel. The first is a safety clamp that normal images do not approach; the second selects which period the stage smooths with rather than whether it runs.

## Highlight recovery removed

- **Was:** a final stage restored near-white luminance that processing had removed, gated from source Y `0.90` to `0.99`.
- **Now:** removed. Measurement showed the gate caught almost none of the luminance actually lost in bright areas, because the loss occurs well below `0.90` where very little of an image crosses that threshold. The loss was global tone compression rather than a highlight-specific artifact, and the new Tone restoration stage repairs it across the whole tonal range.
- The `--highlight-recovery-strength`, `--highlight-recovery-start` and `--highlight-recovery-full` options and the GUI **Highlight recovery** control are gone. **Settings JSON files containing `highlight_recovery_strength`, `highlight_recovery_start` or `highlight_recovery_full` will be rejected on import.** Remove those three keys, or re-export the recipe from this version.

## PWM / Step residual cleanup

- Reworked **PWM / Step** residual cleanup from the early single-template prototype into a hierarchy of robust timing and fitting paths for straight rolling-shutter PWM bands.
- Added a **fundamental-first Auto-period detector** for PWM residuals. Scene-resistant repeated transition evidence is autocorrelated, but near-tied `P / 2P / 3P / ...` plateaus no longer select an arbitrary large lag: the detector prefers the shortest strong recurrent candidate that still has enough visible cycles and two opposite validated PWM transitions. Manual Profile periods remain exact.
- The shortest lag in the search range is now reachable. Previously it was scored but could never be selected as a local maximum, so the correct period was structurally unavailable whenever it sat at the search floor. Parabolic sub-sample refinement also works at the floor.
- Added **phase-slope period refinement**. The coarse detector is limited by its sampling grid, which is too imprecise for images with many visible cycles: at 20 or more cycles a period error under one percent is enough to lose most of the correction. The refinement fits phase drift across the frame and is accepted only when it explains more coherent band energy than the coarse estimate, so a weak fit on a frame with few usable blocks cannot degrade an already-good value.
- Refinement is no longer gated on a whole-frame amplitude measurement. That measurement collapses toward zero as cycle count rises and period error accumulates, so images with many cycles could never qualify for the refinement that would have fixed their period.
- Added image-side PWM timing so the residual profile can work when Restormer contributes little or no useful PWM-frequency correction. Cumulative Restormer source-to-output Y/CbCr changes remain available as optional timing/validation evidence rather than mandatory correction magnitude.
- Added **single-source phase-lock** logic: when the same fundamental phase is strongly coherent across widely separated scene strips, period/phase are frozen globally and coherent radiometric regions may adapt amplitude without inventing independent local phases.
- Added **multi-period / multi-surface PWM fitting** for scenes with evidence for multiple LED timing families. Harmonic-related candidates are grouped to avoid mistaking one square wave's harmonics for separate sources; independent validated periods are fit jointly with radiometric region support.
- Added cycle-consensus and held-out surface-conditioned fallback fits so local PWM authority can shrink/revert when a surface model fails to predict recurrent bands safely.

## Final PWM polish

- Added an optional **Final PWM polish** after the main residual/local cleanup. It searches for no new frequency: it reuses only period families already validated by the main PWM stage, measures the remaining exact PWM-mode energy, performs a bounded authority search, and guards nearby control frequencies.
- The polish stops when the remaining residual is **no longer phase-coherent** with the band. Energy magnitude alone cannot distinguish a residual still in phase with the band from one already driven past zero and inverted, so a magnitude-based stop keeps reporting progress after the band is gone and buys it by overshooting. Coherence makes that distinction, and it was already being computed as an accept/reject gate.
- Because the stage stops on evidence, **PWM polish max passes** is an upper bound rather than a value that needs tuning per image. Range `1-6`.
- New GUI/CLI controls are `flat_profile_pwm_polish`, `flat_profile_pwm_polish_strength`, and `flat_profile_pwm_polish_passes`. Polish controls grey out whenever **Smooth periodic** is selected; switching back to **PWM / Step** restores normal availability without discarding the stored checkbox state.

## Restormer stage

- Added **Enable Restormer correction**. With Restormer disabled, neural model files are not loaded or required and deterministic cleanup runs directly from the source image. CLI equivalent: `--no-restormer`.
- Changed the **desktop GUI default** so Restormer starts **off** for new images and after resetting processing settings. The CLI keeps Restormer enabled by default for backward compatibility.
- Added separate **First-pass luminance strength** and **First-pass chroma strength** controls (`--first-pass-luma-strength`, `--first-pass-chroma-strength`, both default 1.0, range 0-2). Pass 2 remains controlled independently by the existing **Second-pass strength**.
- Renamed the GUI pass-1 labels to **First-pass …** so their wording matches **Second-pass strength**.
- Restormer-only settings now grey out when the Restormer checkbox is off, matching the other optional sections. Band direction remains editable because deterministic cleanup still needs the orientation. AMP remains enabled only when Restormer is on and the selected device may use CUDA.

## Broad residual cleanup

- Added and refined **Multi-surface consensus** for very broad/few-cycle residuals shared across unrelated surfaces, including sub-cycle/single-trough luminance recovery, cross-region agreement, bounded no-harm authority, vector-aware chroma validation, and one optional low-authority same-waveform luminance refinement.
- Changed the default Broad mode for new settings from `dominant` to **`consensus` / Multi-surface consensus**. `dominant` remains available explicitly, and existing JSON files that already save `flat_surface_equalizer_mode: dominant` keep that choice.
- With Restormer enabled, consensus can use cumulative neural Y/CbCr changes as validation evidence. With Restormer disabled, luminance consensus falls back to image-only cross-region agreement; consensus chroma remains conservative without a neural chroma-direction hint.

## Settings export and import

- Added the Restormer, PWM-polish and tone-restoration values to the complete developer settings export; painted cleanup mask pixels remain session/image data and are still excluded from JSON.
- Renamed the developer utility group to **Export/import settings (dev)** and added an **Import json** button directly below **Export json**.
- Added validated settings import for Flicker Suppressor processing JSON. Imports reject malformed/non-object JSON, unknown keys, wrong value types, invalid enum choices, non-finite or out-of-range numbers, malformed RGB cutoffs, and invalid Shadow/Highlight ordering before changing the current image.
- Older/partial settings exports remain usable when they contain only known processing keys: missing fields are filled from current defaults and hidden parser policy is normalized through the same inference namespace used by export.
- Import applies to the current activated image, preserves authored per-stage paint masks, and marks the cached preview stale so Preview/Export recomputes with the imported recipe. Mask pixels remain intentionally outside developer JSON.

---

# v1.0.0 — 2026-08-17

- First public version.

---

*Last Updated: 24 August 2026*
