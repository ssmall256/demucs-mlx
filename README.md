# demucs-mlx

Music source separation optimized for Apple Silicon, powered by [MLX](https://github.com/ml-explore/mlx). A pure-MLX fork of [Demucs](https://github.com/facebookresearch/demucs) with custom Metal kernels for maximum throughput.

## Features

- **~67x realtime** on Apple Silicon — 2.4x faster than Demucs with PyTorch MPS
- Custom fused Metal kernels (GroupNorm+GELU, GroupNorm+GLU, OLA)
- Metal-free fallbacks for non-Apple platforms (Linux)
- No PyTorch required at inference time
- Audio I/O via [mlx-audio-io](https://github.com/ssmall256/mlx-audio-io)
- STFT/iSTFT via [mlx-spectro](https://github.com/ssmall256/mlx-spectro)

## Requirements

- Python >= 3.10
- macOS with Apple Silicon (recommended) or Linux with MLX

## Install

```bash
pip install demucs-mlx
```

Or with uv:

```bash
uv sync
```

For development extras:

```bash
uv sync --extra dev
```

## CLI usage

```bash
demucs-mlx /path/to/audio.wav
```

Options:

```
-n, --name          Model name (default: htdemucs)
-o, --out           Output directory (default: separated)
--shifts            Number of random shifts (default: 1)
--overlap           Overlap ratio (default: 0.25)
-b, --batch-size    Batch size (default: 4)
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

## Performance

Benchmarked on a 3:15 stereo track (44.1 kHz, 16-bit) using `htdemucs` with default settings:

| Package | Backend | Time | Speedup |
|---------|---------|------|---------|
| `demucs` 4.0.1 | PyTorch (CPU) | 52.3s | 0.1x |
| `demucs` 4.0.1 | PyTorch (MPS) | 6.9s | 1x |
| `demucs-mlx` 1.0.0 | MLX + Metal | 2.9s | **2.4x** |

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

Pre-converted MLX weights are cached under `~/.cache/demucs-mlx`. Delete to force re-conversion.

## Documentation

- API reference: `docs/api.md`
- Development workflow: `docs/development.md`
- Platform notes: `docs/platform.md`

## License

MIT. See `LICENSE` for details.
