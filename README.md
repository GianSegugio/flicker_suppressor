# **Flicker Suppressor** 🎨

Single-image/batch restoration for rolling-shutter flicker/banding under temporally modulated artificial lighting. This tool combines a Restormer-based luminance correction estimator, a dedicated two-channel chroma branch, orientation-aware processing, deterministic cleanup for difficult residual bands, and optional per-stage GUI paint masks.

Flicker Suppressor runs locally. Images are not uploaded anywhere by the program.

The project includes a PySide6 desktop application for interactive single-image and batch workflows, plus the full command-line interface for scripting, batch processing, and advanced parameters.
![Flicker Suppressor: desktop application.](img/screenshots/GUI_01.png)

## Example

**Severe rolling bands caused by PWM LED driver:**
![Flicker Suppressor: bands are correctly removed and original colors are restored.](img/examples/example_01.png)

## What it is designed to fix

Flicker Suppressor targets spatial rolling-shutter artifacts caused by artificial lights whose output changes while the sensor is being read, including:

- horizontal or vertical brightness bands;
- green/magenta/blue/yellow chromatic band shifts;
- severe high-contrast periodic bands;
- faint residual row/column banding after the neural restoration pass;
- broad/few-cycle residual bands on large surfaces.

It is **not** intended as a general JPEG-debander, posterization remover, moire remover, fixed-pattern-noise remover, or temporal video-deflicker tool.

## System requirements

### Tested configuration

The current release has been developed and validated primarily on Windows 11 25H2 with:

- Python 3.12;
- PyTorch 2.9.1 + CUDA 12.8 wheel (`torch==2.9.1+cu128`);
- NVIDIA RTX 5090 Mobile GPU;
- `numpy==1.26.4`;
- `Pillow==10.4.0`;
- `einops==0.8.0`;
- `PySide6==6.8.3` for the desktop GUI.

A recent NVIDIA driver is required for CUDA GPU mode. CPU mode is supported but substantially slower. The code also contains an Apple MPS path and is likely portable to ROCm builds that expose the PyTorch `torch.cuda` interface, but those paths have not been validated to the same level as NVIDIA CUDA. Intel XPU is not supported by this release.

The portable Windows build produced by `build_windows.ps1` bundles Python, Qt/PySide6, PyTorch, the model files, and the required runtime libraries; an end user of that standalone folder does not need a separate Python or Qt installation.

## Desktop GUI dev mode

For source/development use on Windows:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_gui_dev.ps1
```

The launcher creates `.venv-gui` when needed, installs the tested CUDA PyTorch build plus the GUI requirements, and starts `gui_main.py`.

On startup, Flicker Suppressor shows a lightweight centered splash before importing the heavier GUI/PyTorch/CUDA application modules. It uses the application logo with a separate bold white **`loading...`** label below it and a blurred dark text shadow for readability over bright desktop content. The splash has no artificial minimum duration and closes as soon as the main window is ready.

## Build

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build_windows.ps1
```

The standalone folder is produced under `build\gui_main.dist`. See `GUI_BUILD.md` for the build workflow.

## Testing in Python

Create a virtual environment if desired:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### NVIDIA GPU - tested PyTorch build

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r .\requirements.txt
```

`torchvision` and `torchaudio` are not required by Flicker Suppressor.

CUDA FP16 autocast (AMP) is enabled by default when the resolved device is CUDA. Use `--no-amp` for an FP32 comparison. On CPU/MPS the AMP flag has no effect.

On multi-GPU systems, choose a specific visible NVIDIA GPU with an indexed device such as `--device cuda:0` or `--device cuda:1`. The desktop Device dropdown enumerates CUDA GPUs by model name and labels the hardware selected by Auto.

### CPU-only

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r .\requirements.txt
```

Verify PyTorch:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## CLI quick start

Basic one-pass restoration:

```powershell
python .\hybrid_infer_detail_preserving.py `
    --input .\photo.jpg `
    --output .\photo_restored.png `
    --luma-model .\models\y.pth `
    --chroma-model .\models\chroma.pth `
    --device cuda `
    --amp
```

`--band-axis auto` is the default. Output is written as PNG.

### Strong/severe bands

Use a second pass when one pass leaves obvious residual bands:

```powershell
python .\hybrid_infer_detail_preserving.py `
    --input .\photo.jpg `
    --output .\photo_restored.png `
    --luma-model .\models\y.pth `
    --chroma-model .\models\chroma.pth `
    --passes 2 `
    --flat-filter `
    --device cuda `
    --amp
```

### Faint residual bands on textured surfaces

```powershell
python .\hybrid_infer_detail_preserving.py `
    --input .\photo.jpg `
    --output .\photo_restored.png `
    --luma-model .\models\y.pth `
    --chroma-model .\models\chroma.pth `
    --flat-profile `
    --flat-profile-luma-strength 0.50 `
    --flat-profile-chroma-strength 0.50 `
    --device cuda `
    --amp
```

For exceptionally strong fine periodic bands, values around `0.8-1.2` can be useful. Increase gradually. Values above `1.0` can intentionally over-apply the estimated residual correction on textured/foreground regions; when Flat-region cleanup is also active, the dominant-surface handoff caps that over-application on the main large smooth background surface.

Residual-profile correction is spatially adaptive by default: Flicker Suppressor still estimates one robust global band period/phase/waveform, then fits only a slowly varying local amplitude from the band-limited residual that matches that waveform. The current default is attenuation-only (`max gain = 1.0`), which lets weakly supported regions receive less correction without silently exceeding the user's selected profile strength. The no-harm validator is band-coherent: it may vary only along the band direction, so it can reject a harmful scene zone without drawing 2-D object silhouettes into the profile correction.

### Very broad residuals on one dominant surface

```powershell
--flat-surface-equalizer
```

This is an advanced, stronger option for large walls or other dominant surfaces where only a few very broad band cycles remain. See `DOCUMENTATION.md` before enabling it routinely.

## Batch processing

`--input` may be a single image or a directory. Directory input is searched recursively for:

`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`.

Example:

```powershell
python .\hybrid_infer_detail_preserving.py `
    --input .\input_photos `
    --output .\restored_photos `
    --luma-model .\models\y.pth `
    --chroma-model .\models\chroma.pth `
    --device cuda `
    --amp
```

The source directory structure is preserved and all outputs are written as PNG. Existing outputs are skipped unless `--overwrite` is used.

## Recommended presets

| Situation | Suggested options |
|---|---|
| Normal case | defaults; CUDA AMP is on automatically (`--no-amp` forces FP32) |
| Severe bands | `--passes 2 --flat-filter` |
| Faint residual bands / textured wall | `--flat-profile --flat-profile-luma-strength 0.50 --flat-profile-chroma-strength 0.50` (add `--flat-filter` only when local flat-surface cleanup is also useful) |
| Very fine strong periodic bands | previous preset + raise profile strengths; optionally force `--flat-profile-band-period` |
| Visible vertical bands | `--band-axis vertical` or Auto |
| Mixed horizontal + vertical residuals | add `--orthogonal-profile`; it is independent of the main `--flat-profile` switch |
| Very broad few-cycle residual on dominant surface | `--flat-surface-equalizer` can be enabled independently; combine with other cleanup stages only when needed |
| Localized cleanup in the GUI | paint the corresponding Flat / Residual / Broad **Mask**; the mask gates only that stage's visible correction |
| Debugging masks/period detection | add `--debug-dir .\debug` |

## Documentation

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for:

- desktop GUI workflow, per-image activation, selection/current-image semantics, per-stage paint masks, export behavior, and settings validation;
- architecture and processing flow;
- Y/CbCr separation;
- correction-field mathematics;
- orientation handling;
- flat-filter/profile/equalizer internals;
- detailed explanation of **every inference command-line option**;
- troubleshooting and tuning examples.

## Limitations

- This is a single-image restoration system; it cannot use temporal information from adjacent video frames.
- Very broad real illumination gradients can be mathematically ambiguous with extremely low-frequency flicker.
- Auto orientation uses a physical aspect-ratio prior for ordinary portrait/landscape images. Manually rotated/cropped files can require `--band-axis horizontal` or `vertical`.
- Aggressive local flattening can alter legitimate smooth surfaces if safety thresholds are relaxed too far.
- Aggressive profile strengths can overcorrect real row/column illumination variation; the high-strength guards reduce common failures but cannot prove that every scene variation is flicker.
- Painted GUI cleanup masks are session/per-image editing data; they are used for preview/export but are not serialized into the developer settings JSON.
- Current GUI and CLI exports are 8-bit RGB PNG; original RAW/high-bit-depth image metadata is not preserved.

## License and third-party software

Flicker Suppressor is released under the Apache License 2.0. See `LICENSE`.

Third-party notices are kept separately for development/source use and for the frozen Windows release:

- `LICENSE-THIRD-PARTY` - development/source-tree third-party notices;
- `RELEASE-LICENSE-THIRD-PARTY` - release-specific notices and license texts for the bundled Windows portable build.

Important upstream projects:

- Restormer: https://github.com/swz30/Restormer - MIT License.
- BurstDeflicker: https://github.com/qulishen/BurstDeflicker - Apache License 2.0.
- BasicSR: https://github.com/XPixelGroup/BasicSR - Apache License 2.0.

## Research attribution

BurstDeflicker:

```bibtex
@inproceedings{BurstDeflicker_lishenqu,
  title={BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes},
  author={Qu, Lishen and Liu, Zhihao and Zhou, Shihao and Luo, Yaqi and Liang, Jie and Zeng, Hui and Zhang, Lei and Yang, Jufeng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

Restormer:

```bibtex
@inproceedings{Zamir2021Restormer,
  title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
  author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat and Fahad Shahbaz Khan and Ming-Hsuan Yang},
  booktitle={CVPR},
  year={2022}
}
```

---

*Last Updated: 17 August 2026*
