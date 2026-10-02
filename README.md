# demucs-mlx

Split any song into its individual stems — vocals, drums, bass, and other instruments — directly on your Mac.

demucs-mlx is a fast, native Apple Silicon port of Meta's [Demucs](https://github.com/adefossez/demucs) music source separation model, built on [MLX](https://github.com/ml-explore/mlx). No PyTorch required.

## Features

- **Up to 94.8x realtime** on Apple Silicon (M4 Max, measured with the now opt-in fp16 attention; fp32, the default, is ~4% slower) — >3.3x faster than Demucs with PyTorch MPS
- **Auto-tuned hardware topology** — automatically configures batch sizes to match Apple Silicon memory bandwidth, GPU cores, and SLC cache
- **Matches upstream Demucs** for every registry model: 72–83 dB SNR per stem against PyTorch on the same input, checked in CI
- Custom fused Metal kernels (GroupNorm+GELU, GroupNorm+GLU, zero-transpose GLU, OLA)
- Metal-free fallbacks for non-Apple platforms (Linux)
- No PyTorch required at inference time
- Automatic resampling — input files at any sample rate are resampled to the model rate
- Audio I/O via [mlx-audio-io](https://github.com/ssmall256/mlx-audio-io)
- STFT/iSTFT via [mlx-spectro](https://github.com/ssmall256/mlx-spectro)

## Requirements

- Python >= 3.10
- macOS with Apple Silicon (recommended) or Linux with MLX
- MLX 0.31.2 to 0.32.x, with mlx-audio-io 1.3.x and mlx-spectro 0.9.3 or newer

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
-b, --batch-size    Batch size (default: auto, matched to hardware topology)
--compile           Opt in to whole-forward compilation for fixed shapes
--attention         Attention kernel precision: fp16 (default) or fp32 (~3% slower, same accuracy)
--write-workers     Concurrent writer threads (default: 2)
--ane-time-encoder  Offload the first HTDemucs waveform convolution to the Neural Engine
--stem              For htdemucs_ft, compute only drums, bass, other, or vocals
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

## Tuning

**The defaults are the recommended configuration.** You should not need to set
anything to get the best results; this section exists so the levers are
discoverable rather than buried in source.

| Setting | Default | Why |
|---|---|---|
| `-b` / `--batch-size` | `auto` | Measured per machine: 3 on M4 Pro and 32-core M4 Max, 8 on 40-core M4 Max with >= 64 GB, 2 elsewhere. Chunks are spread evenly over the batches the target implies. |
| `--attention` | `fp16` | Runs only the attention kernel in half precision; the projections stay fp32. Within 0.5 dB of `fp32` against upstream and ~3% faster end to end. `DEMUCS_MLX_ATTENTION_FP16=0` also selects `fp32`. |
| `--compile` | `None` (off) | Compiles repeated forward graph execution blocks for fixed chunk shapes. |
| `--shifts` | `1` | Matches upstream Demucs. Each extra shift costs a full pass. |
| `--overlap` | `0.25` | Matches upstream Demucs. |
| `--write-workers` | `2` | Encodes stems concurrently while the next track runs. |

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `DEMUCS_MLX_USE_FUSED_GN_GLU` | `0` (off) | Runs GroupNorm+GELU/GLU through fused Metal kernels instead of the pure-MLX path. Output agrees with the unfused path to 118 dB SNR and is deterministic run to run. Timing is a wash on current hardware — 1.7331 s against a 1.7270 s control at a 1.76% noise floor — so the unfused path stays the default. Both paths expose identical parameter names, so an existing converted cache loads either way. |
| `DEMUCS_MLX_COMPILE_FORWARD` | `1` (on) | Compiles repeated GPU forward shapes after their first eager call (or pass `--no-compile`). Nested DConv compilation is automatically suppressed to allow global kernel fusion across the full graph. The ANE path always bypasses this compilation. See [throughput experiments](docs/throughput.md). |
| `DEMUCS_MLX_COMPILE_DCONV` | `auto` (`1` eager, `0` compiled) | Compile DConv inference blocks after weights load. Automatically defaults to `0` when outer forward compilation is active, and `1` otherwise. Set explicitly to override. |

## Version history

See [CHANGELOG.md](CHANGELOG.md), which the release workflow reads directly.

## Performance

Benchmarked on a 3:15 stereo track (44.1 kHz, 16-bit) using `htdemucs` with default settings:

| Package | Backend | Time | Speedup |
|---------|---------|------|---------|
| `demucs` 4.0.1 | PyTorch (CPU) | 52.3s | 0.1x |
| `demucs` 4.0.1 | PyTorch (MPS) | 6.9s | 1x |
| `demucs-mlx` 1.1.0 | MLX + Metal | 2.7s | **2.6x** |

*Apple M4 Max, 128 GB. All runs use `htdemucs` with default settings and a single warm-up pass before timing.*

In a direct alternating comparison, the current development branch's three GPU optimizations together reduced default `htdemucs` separation time by **24–27%**, increasing audio throughput by **31–38%**. These are synthetic-input, loaded-model measurements that exclude file I/O; the comparison job had substantial background GPU use, so its absolute times should not be compared with the track benchmark above. See the [reproducible throughput measurements](docs/throughput.md) for the paired results and fidelity.

## Models

| Model | Sources | Description |
|-------|---------|-------------|
| `htdemucs` | 4 | Hybrid Transformer Demucs (default) |
| `htdemucs_ft` | 4 | Fine-tuned HTDemucs |
| `htdemucs_6s` | 6 | 6-source (adds piano, guitar) |
| `hdemucs_mmi` | 4 | Hybrid Demucs MMI |
| `mdx` | 4 | Music Demixing model |
| `mdx_extra` | 4 | MDX with extra training |

For a single fine-tuned stem, `--stem` runs only its specialized model:

```bash
demucs-mlx -n htdemucs_ft --stem vocals song.wav
demucs-mlx -n htdemucs_ft --stem drums song.wav
demucs-mlx -n htdemucs_ft --stem bass song.wav
```

The full four-stem `htdemucs_ft` run remains available by omitting `--stem`.

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
- Neural Engine waveform prototype and measured results: `docs/ane-prototype.md`
- Throughput experiments and GroupNorm speedup: `docs/throughput.md`

## License

MIT. Based on [Demucs](https://github.com/adefossez/demucs) by Meta Research. See `LICENSE` for details.
