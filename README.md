# demucs-mlx

Split any song into its individual stems — vocals, drums, bass, and other instruments — directly on your Mac.

demucs-mlx is a fast, native Apple Silicon port of Meta's [Demucs](https://github.com/adefossez/demucs) music source separation model, built on [MLX](https://github.com/ml-explore/mlx). No PyTorch required.

For Swift, macOS and iOS apps, see [demucs-mlx-swift](https://github.com/ssmall256/demucs-mlx-swift), the native Swift
package that shares the same models.

## Features

- **114x realtime** once warm on a 40-core M4 Max with default settings — 2.7x faster than stock Demucs on PyTorch MPS (2.3x if Demucs is patched to keep its STFT on the GPU) and 18x faster than PyTorch on CPU ([measurements](#performance))
- **Auto-tuned hardware topology** — automatically configures batch sizes to match Apple Silicon memory bandwidth, GPU cores, and SLC cache
- **Matches upstream Demucs** for every registry model: 72–83 dB SNR per stem against PyTorch on the same input, checked in CI
- Custom fused Metal kernels (GroupNorm+GELU, GroupNorm+GLU, zero-transpose GLU, OLA)
- No PyTorch required at inference time
- Automatic resampling — input files at any sample rate are resampled to the model rate
- Audio I/O via [mlx-audio-io](https://github.com/ssmall256/mlx-audio-io)
- STFT/iSTFT via [mlx-spectro](https://github.com/ssmall256/mlx-spectro)

## Requirements

- Python >= 3.10
- macOS on Apple Silicon. The code has fallbacks for MLX without Metal, but Linux is not tested.
- MLX 0.32.x (0.32.3 or newer), with mlx-audio-io 1.3.24 or newer and mlx-spectro 0.9.10 or newer

## Install

```bash
pip install demucs-mlx
```

That is all: the first time you use a model, demucs-mlx downloads its MLX
weights from [Hugging Face](https://huggingface.co/ssmall256/demucs-mlx) into
`~/.cache/demucs-mlx` (160 MB for the default `htdemucs`). The download is
accepted only if it matches the SHA-256 digest built into this package, and
PyTorch is never needed.

To work offline, set `DEMUCS_MLX_NO_DOWNLOAD=1` (or `HF_HUB_OFFLINE=1`) and
either fetch the files ahead of time:

```bash
python -m demucs_mlx.hub download htdemucs --output-dir ~/.cache/demucs-mlx
```

or convert them yourself from Meta's official checkpoints, which needs the
conversion extra:

```bash
pip install 'demucs-mlx[convert]'
python -m demucs_mlx.mlx_convert htdemucs --output-dir ~/.cache/demucs-mlx
```

Conversion compares every converted model against its PyTorch source before
writing the cache and reproduces the published files byte for byte. It is also
the automatic fallback if a download fails. `HF_ENDPOINT` selects a mirror. If
the optional `mlx-weights` package is installed, demucs-mlx uses its shared
cache directory instead of `~/.cache/demucs-mlx`.

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
--compile / --no-compile  Compile the GPU forward per chunk shape (default: on)
--attention         Attention kernel precision: fp16 (default) or fp32 (~3% slower, same accuracy)
--prefetch-tracks   Decoded input prefetch depth (default: 2)
--write-workers     Concurrent writer threads (default: 4)
--io-memory-mib     Overlapping I/O budget (ceiling: min(512 MiB, RAM/8); 0: serial)
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
| `--compile` / `--no-compile` | unset (follows `DEMUCS_MLX_COMPILE_FORWARD`, on) | Compiles the GPU forward for each chunk shape. |
| `--shifts` | `1` | Matches upstream Demucs. Each extra shift costs a full pass. |
| `--overlap` | `0.25` | Matches upstream Demucs. |
| `--write-workers` | `4` | Encodes stems concurrently while the next track runs. FLAC encoding scales with workers: 4 stems of a 3:15 track take 0.62 s at 4 against 1.22 s at 2 (M4 Max). |

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `DEMUCS_MLX_USE_FUSED_GN_GLU` | `0` (off) | Runs GroupNorm+GELU/GLU through fused Metal kernels instead of the pure-MLX path. Output agrees with the unfused path to 118 dB SNR and is deterministic run to run. Timing is a wash on current hardware — 1.7331 s against a 1.7270 s control at a 1.76% noise floor — so the unfused path stays the default. Both paths expose identical parameter names, so an existing converted cache loads either way. |
| `DEMUCS_MLX_COMPILE_FORWARD` | `1` (on) | Compiles each GPU forward shape on its first call, so the first call returns exactly what later calls do. A valid-length tail chunk (non-HTDemucs models), which rarely recurs, runs eagerly and compiles only if seen again. `0` or `--no-compile` keeps every forward eager. Nested DConv compilation is automatically suppressed to allow global kernel fusion across the full graph. See [throughput experiments](docs/throughput.md). |
| `DEMUCS_MLX_COMPILE_DCONV` | `auto` (`1` eager, `0` compiled) | Compile DConv inference blocks after weights load. Automatically defaults to `0` when outer forward compilation is active, and `1` otherwise. Set explicitly to override. |

## Throughput

RTFx is audio seconds divided by wall seconds. Measured on an M4 Max (MLX 0.32.3),
htdemucs reaches 114× once warm. The first separation in a new process is
slower, about 99–107×, because it compiles the model graph, prepares Metal
kernels and allocates GPU memory.

For the best sustained throughput:

1. **Keep one process and one `Separator` alive** and pass it every file (the CLI
   does this for all files given in one invocation). Starting a process per file
   pays Python start-up, model loading and the slower first call every time.
2. **Leave `--batch-size` at `auto`.** It spreads chunks evenly across batches,
   so a track compiles and allocates for fewer distinct shapes. On a 120-second
   track the first call took 1.25 s with `auto` against 1.39–1.46 s at batch 8.
3. **Do not clear MLX's buffer cache between tracks.** Later calls reuse it; a call
   after `mx.clear_cache()` was 75–135 ms slower. MLX keeps roughly 16–31 GB
   cached with htdemucs and 64–83 GB with the `mdx` models on a 128 GB machine.
   On smaller machines, `mx.set_cache_limit()` caps it at the cost of
   re-allocating.
4. **Close other GPU-heavy apps.** Anything else rendering or computing on the GPU
   shares it; measurements slowed to 70× and below while other apps were busy.
5. **Check long sessions for slowdown** with
   `python benchmarks/bench_rtfx.py --audio song.m4a --processes 1 --calls 30`: if the
   last-five median falls below the warm median, the machine is throttling.

`python benchmarks/bench_rtfx.py --audio song.m4a --processes 3 --calls 10` reports
first-call, warm and sustained RTFx in fresh processes; its docstring defines
each measurement.

## Version history

See [CHANGELOG.md](CHANGELOG.md), which the release workflow reads directly.

## Performance

`htdemucs`, default settings, 120 seconds of stereo 44.1 kHz audio, tensor in to
stems out, median of four warmed calls in alternating order:

| Package | Backend | Time | Realtime factor |
|---------|---------|------|---------|
| `demucs` 4.1.0, PyTorch 2.14.1 | CPU | 19.4 s | 6x |
| `demucs` 4.1.0, PyTorch 2.14.1 | MPS | 2.82 s | 42x |
| same, patched to keep STFT/iSTFT on MPS | MPS | 2.39 s | 50x |
| `demucs-mlx` (this release), MLX 0.32.3 | MLX + Metal | 1.05 s | **114x** |

*Apple M4 Max (40-core GPU), 128 GB, October 2026. File decoding, model loading
and the first call in a process are excluded; see [Throughput](#throughput) for
those. Stock Demucs moves its STFT and iSTFT to the CPU when the model is on MPS;
the patched row keeps them on the GPU, which changes its output by less than
-120 dB. Other machines will differ. [docs/throughput.md](docs/throughput.md)
records the individual optimizations and how they were measured.*

## Models

| Model | Sources | Description |
|-------|---------|-------------|
| `htdemucs` | 4 | Hybrid Transformer Demucs (default) |
| `htdemucs_ft` | 4 | Fine-tuned HTDemucs |
| `htdemucs_6s` | 6 | 6-source (adds piano, guitar) |
| `hdemucs_mmi` | 4 | Hybrid Demucs MMI |
| `mdx` | 4 | Music Demixing model |
| `mdx_extra` | 4 | MDX with extra training |
| `mdx_q`, `mdx_extra_q` | 4 | The same bags from Meta's quantized checkpoints (conversion needs `diffq`) |

For a single fine-tuned stem, `--stem` runs only its specialized model:

```bash
demucs-mlx -n htdemucs_ft --stem vocals song.wav
demucs-mlx -n htdemucs_ft --stem drums song.wav
demucs-mlx -n htdemucs_ft --stem bass song.wav
```

The full four-stem `htdemucs_ft` run remains available by omitting `--stem`.

## MLX model cache

MLX weights are cached under `~/.cache/demucs-mlx` by default. Set
`DEMUCS_MLX_CACHE_DIR` to use another directory. When the optional `mlx-weights`
package is installed and that variable is unset, demucs-mlx uses its shared
`~/.cache/mlx-weights/demucs-mlx` directory instead.

The cache is a stable, shared location: other tools may read it or fill it. It
holds exactly the files published at
[ssmall256/demucs-mlx](https://huggingface.co/ssmall256/demucs-mlx), so a
directory populated by a download, by local conversion or by another tool is
interchangeable, and [demucs-mlx-swift](https://github.com/ssmall256/demucs-mlx-swift) reads the same files.

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

Downloaded weights are safetensors files, which hold arrays and no code. Each
file's size and SHA-256 are fixed in `demucs_mlx/hub.py`; a download that
differs is discarded before anything is written to the cache, and the loader
then validates the config and re-checks the weight digest as it does for a
local conversion. `python -m demucs_mlx.hub verify <directory>` checks a
directory against those digests.

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
- Throughput measurements and how to reproduce them: `docs/throughput.md`

## Differences from upstream Demucs

- Inference only; there is no training code.
- `Separator` does not support progress callbacks or multi-process `jobs`.
- Input at any sample rate is resampled to the model rate automatically.
- PyTorch is never needed to run a model. It is only needed to convert weights
  yourself instead of downloading them.

## License

MIT. Based on [Demucs](https://github.com/adefossez/demucs) by Meta Research. See `LICENSE` for details.
