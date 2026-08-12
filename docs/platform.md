# Platform notes

This project is uv-first. All instructions below use uv and the `uv.lock` workflow.

## macOS

- Apple Silicon is supported via MLX.
- Audio I/O is handled natively by mlx-audio-io (no FFmpeg required).
- The current native runtime pair is MLX 0.31.2 with mlx-audio-io 1.3.11. MLX 0.32 requires a future matching audio-I/O wheel.

Typical flow:

```bash
uv lock
uv sync
uv run demucs-mlx /path/to/audio.wav
```

## Linux

- Python >= 3.10 required.

```bash
uv lock
uv sync
uv run demucs-mlx /path/to/audio.wav
```

## Windows

Not supported. MLX requires macOS or Linux.
