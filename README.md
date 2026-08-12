# demucs-mlx

Split any song into its individual stems — vocals, drums, bass, and other instruments — directly on your Mac.

demucs-mlx is a fast, native Apple Silicon port of Meta's [Demucs](https://github.com/adefossez/demucs) music source separation model, built on [MLX](https://github.com/ml-explore/mlx). No PyTorch required.

## Features

- **~73x realtime** on Apple Silicon — 2.6x faster than Demucs with PyTorch MPS
- **Bit-exact parity** with upstream Demucs stems (within floating-point tolerance)
- Custom fused Metal kernels (GroupNorm+GELU, GroupNorm+GLU, OLA)
- Metal-free fallbacks for non-Apple platforms (Linux)
- No PyTorch required at inference time
- Automatic resampling — input files at any sample rate are resampled to the model rate
- Audio I/O via [mlx-audio-io](https://github.com/ssmall256/mlx-audio-io)
- STFT/iSTFT via [mlx-spectro](https://github.com/ssmall256/mlx-spectro)

## Requirements

- Python >= 3.10
- macOS with Apple Silicon (recommended) or Linux with MLX
- MLX 0.31.2 paired with mlx-audio-io 1.3.11; the native audio package does not yet support MLX 0.32

## Install

```bash
pip install demucs-mlx
```

On first run, demucs-mlx loads cached MLX weights if available. If the optional
`mlx-weights` package is installed locally, demucs-mlx uses its shared cache. Otherwise
it uses its built-in cache. A cache miss is converted internally from the official
Demucs registry using the restricted loader described below.

To bootstrap a missing model with the public package, install the conversion extra:

```bash
pip install 'demucs-mlx[convert]'
```

You can explicitly generate a safe cache in any directory with:

```bash
python -m demucs_mlx.mlx_convert htdemucs --output-dir ~/.cache/demucs-mlx
```

Once weights are cached, the `convert` extra is no longer needed for inference.

## CLI usage

```bash
demucs-mlx /path/to/audio.wav
```

Options:

```
-n, --name          Model name (default: htdemucs)
-o, --out           Output directory (default: separated)
--shifts            Number of random shifts (default: 1)
--seed              Optional RNG seed for reproducible shifts (default: none)
--overlap           Overlap ratio (default: 0.25)
-b, --batch-size    Batch size (default: 2)
--write-workers     Concurrent writer threads (default: 1)
--list-models       List available models
-v, --verbose       Verbose logging
```

## Python usage

```python
from demucs_mlx import Separator

separator = Separator()
origin, stems = separator.separate_audio_file("song.wav")

# stems is a dict: {"drums": array, "bass": array, "other": array, "vocals": array}
for name, audio in stems.items():
    print(f"{name}: {audio.shape}")
```

To keep outputs as MLX arrays (avoids GPU-to-CPU copy):

```python
origin, stems = separator.separate_audio_file("song.wav", return_mx=True)
```

For reproducible shift sampling (while keeping `shifts=1` behavior), pass a seed:

```python
separator = Separator(model="htdemucs", shifts=1, seed=0)
origin, stems = separator.separate_audio_file("song.wav")
```

## What changed in 1.4.6

- Restricted official Demucs checkpoint loading to PyTorch 2.6+ with `weights_only=True`, a narrow class allowlist, hash verification, and strict package validation.
- Replaced executable pickle caches with digest-verified MLX safetensors and versioned JSON metadata.
- Legacy pickle caches are never opened; they are ignored while safe artifacts regenerate from the verified official registry.

## What changed in 1.4.5

- Fixed audio prefetch on MLX 0.31.2 by materializing decoded arrays on the producer thread before queue handoff.
- Reduced the default inference batch size from 8 to 2 to avoid memory thrashing on 16–36 GB Macs; explicit `-b` values are unchanged.
- Pinned MLX 0.31.2 and mlx-audio-io 1.3.11 as a compatible native runtime pair. MLX 0.32 support will follow a matching mlx-audio-io release.

## What changed in 1.4.4

- Fixed multi-segment `split=True` overlap-add on MLX 0.31.2. Long inputs no longer produce high-amplitude reconstruction spikes.
- Added regression coverage for split-mode overlap-add and an optional model-level reproduction for issue #1.

## What changed in 1.4.3

- `resample_mx()` now uses direct `mac.resample()` instead of writing/reading a temp file — eliminates an unnecessary MLX→numpy→disk→MLX round-trip.
- Bumped minimum `mlx-audio-io` to `>=1.3.9` (auto-selects best resampling quality).

## What changed in 1.4.2

- Audio loading now stays as native MLX arrays end-to-end (no numpy round-trip).
- Automatic resampling via `mlx-audio-io` — input files no longer need to match the model sample rate.
- Uses `soxr_vhq` resampling quality when available, with automatic fallback.
- Bumped minimum dependencies: `mlx>=0.31.0`, `mlx-audio-io>=1.3.8`, `mlx-spectro>=0.2.4`.

## What changed in 1.4.0

- Fixed shifted-inference `TensorChunk` propagation so chunk length/offset is handled correctly in all paths.
- Added optional deterministic RNG control (`seed`) for Python API and CLI.
- Default behavior is unchanged: `shifts=1` remains stochastic unless `seed` is provided.

## Performance

Benchmarked on a 3:15 stereo track (44.1 kHz, 16-bit) using `htdemucs` with default settings:

| Package | Backend | Time | Speedup |
|---------|---------|------|---------|
| `demucs` 4.0.1 | PyTorch (CPU) | 52.3s | 0.1x |
| `demucs` 4.0.1 | PyTorch (MPS) | 6.9s | 1x |
| `demucs-mlx` 1.1.0 | MLX + Metal | 2.7s | **2.6x** |

*Apple M4 Max, 128 GB. All runs use `htdemucs` with default settings and a single warm-up pass before timing.*

## Models

| Model | Sources | Description |
|-------|---------|-------------|
| `htdemucs` | 4 | Hybrid Transformer Demucs (default) |
| `htdemucs_ft` | 4 | Fine-tuned HTDemucs |
| `htdemucs_6s` | 6 | 6-source (adds piano, guitar) |
| `hdemucs_mmi` | 4 | Hybrid Demucs MMI |
| `mdx` | 4 | Music Demixing model |
| `mdx_extra` | 4 | MDX with extra training |

## MLX model cache

Pre-converted MLX weights are cached under `~/.cache/demucs-mlx` by default. When the
optional `mlx-weights` package is installed, demucs-mlx uses its shared
`~/.cache/mlx-weights/demucs-mlx` directory instead.

Cache format v1 consists of `<model>.safetensors` and a versioned
`<model>_config.json` sidecar. Arrays are saved and loaded with MLX's native
safetensors support. The bounded JSON metadata records the exact MLX model classes and
constructor data, ensemble shape and weights, ordered official Demucs source
signatures/checksums, conversion time, actual MLX version, verification result, and the
SHA-256 of the safetensors file. Exceptional constructor values such as `Fraction` use
a narrowly validated tagged JSON representation. The digest and complete metadata are
validated before arrays are loaded or a model is constructed.

Older `<model>_mlx.pkl` files are unsafe legacy caches. demucs-mlx never opens,
rewrites, or deletes them. If conversion dependencies are installed, a legacy-only
cache is ignored and safe v1 artifacts are regenerated from the verified official
source. Without automatic conversion, the error includes the ignored pickle path and
the exact regeneration command. Partial, corrupt, unversioned, or otherwise invalid
safetensors/config pairs fail closed and never fall back to a pickle; move those safe
artifacts aside and run, for example:

```bash
python -m demucs_mlx.mlx_convert htdemucs --output-dir ~/.cache/demucs-mlx
```

### Model trust boundary

Conversion requires PyTorch 2.6 or newer before any checkpoint is downloaded or
deserialized. Official packages retain filename-hash verification and are loaded with
`weights_only=True` plus a scoped allowlist of exact Demucs classes and narrowly needed
compatibility types. The package shape, exact model class, constructors, and ordinary
or quantized state are validated before trusted Demucs code constructs a model. There
is no unrestricted fallback.

Installed PyTorch, Demucs, NumPy, optional DiffQ quantization code, MLX, and the packaged
official model registry are inside the trust boundary. Arbitrary checkpoint globals
and local pickle caches are not trusted. Restricted loading prevents executable pickle
globals; it is not a resource-exhaustion sandbox for otherwise valid tensor files.

## Documentation

- API reference: `docs/api.md`
- Development workflow: `docs/development.md`
- Platform notes: `docs/platform.md`

## License

MIT. Based on [Demucs](https://github.com/adefossez/demucs) by Meta Research. See `LICENSE` for details.
