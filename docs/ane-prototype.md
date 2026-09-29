# Neural Engine waveform encoder prototype

## Result

The default `htdemucs` waveform encoder converts to a fixed-shape Core ML ML Program, but **Core ML places all 274 operations on the CPU** when loaded with `CPU_AND_NE` on an M4 Max. No public `--ane-time-encoder` switch or `Separator` option is exposed: it would not use the Neural Engine.

The converted diagnostic asset and its weight-identity manifest are saved under `~/.cache/demucs-mlx/ane/`. The manifest records the validated MLX safetensors SHA-256, batch size 2, 343,980 samples (7.8 seconds), and compute placement. The conversion command exits with an explanatory error when the plan contains no Neural Engine operations.

## What was tried

The converter loads the official model through `get_restricted_demucs_model`, wraps waveform normalization and all four `tencoder` stages, traces fixed batch-2 input, and converts with Core ML Tools 9 to FP16 ML Program format. Explicit one-sample padding before stages 2–4 removes TorchScript dynamic shape-to-integer operations that Core ML Tools could not translate. It does not change the official encoder computation.

| Core ML model | Time-axis samples | Preferred compute device |
|---|---:|---|
| Isolated 1D convolution | 2,048 | CPU |
| Isolated 1D convolution | 16,384 | Neural Engine |
| Isolated 1D convolution | 85,995 | CPU |
| Isolated 1D convolution | 343,980 | CPU |
| Isolated 2D convolution with singleton height | 2,048 | CPU |
| Isolated 2D convolution with singleton height | 16,384 | Neural Engine |
| Isolated 2D convolution with singleton height | 85,995 | CPU |
| Isolated 2D convolution with singleton height | 343,980 | CPU |
| Complete waveform encoder | 343,980 | CPU for all 274 operations |

This probe suggests the long time axis is a placement constraint, rather than merely the encoder's use of 1D convolution. The compute plan reports anticipated placement; it does not explain Core ML's scheduling decision. A chunked or reshaped model might be possible, but the encoder's normalization and dilated convolutions make that a separate design requiring parity and throughput work.

The compiled CPU-placed model was compared against MLX with deterministic random stereo input. The table measures the four encoder outputs; the final single-segment batch also passed padding and output-shape checks.

| Encoder stage | SNR against MLX | Peak absolute error |
|---|---:|---:|
| 1 | 56.23 dB | 0.00676 |
| 2 | 46.00 dB | 0.00556 |
| 3 | 42.29 dB | 0.00363 |
| 4 | 39.30 dB | 0.00662 |

These differences include Core ML's FP16 conversion. Full-stem fidelity and alternating 30-/60-second GPU-versus-ANE timings were not run because this model performs **no Neural Engine work**; calling those runs an ANE comparison would be misleading.

## Reproduce

On macOS, install the optional diagnostic dependencies with `uv sync --extra ane --extra ane-convert`. Run the following through MetalQ; the conversion is expected to exit nonzero after saving diagnostic assets because its placement gate fails on the tested host.

```bash
metalq submit -w --no-env-sync -n demucs-ane-convert -- python -m demucs_mlx.ane convert
metalq submit -w --no-env-sync -n demucs-ane-placement -- python tests/probe_ane_placement.py
metalq submit -w --no-env-sync -n demucs-ane-parity -- python tests/probe_ane_parity.py
```

Measurements above used an Apple M4 Max with macOS 27, MLX 0.32.2, Core ML Tools 9.0, and PyTorch 2.7.1. Conversion used the official default `htdemucs` checkpoint and the validated local MLX cache.
