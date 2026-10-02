# Platform notes

This project is uv-first. All instructions below use uv and the `uv.lock` workflow.

## macOS

- Apple Silicon is supported via MLX.
- Audio I/O is handled natively by mlx-audio-io (no FFmpeg required).
- MLX 0.32.x (0.32.3 or newer) is supported, with mlx-audio-io 1.3.23 or newer. Since 1.3.23 mlx-audio-io's extension does not link MLX, so changing the MLX version needs no rebuild.

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
