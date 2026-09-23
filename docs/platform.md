# Platform notes

This project is uv-first. All instructions below use uv and the `uv.lock` workflow.

## macOS

- Apple Silicon is supported via MLX.
- Audio I/O is handled natively by mlx-audio-io (no FFmpeg required).
- MLX 0.31.2 through 0.32.x are supported, with mlx-audio-io 1.3.x. mlx-audio-io ships an sdist whose extension is compiled against whichever MLX the build environment resolves, and refuses to load against a different one, so keep the MLX in your environment and the MLX it was built against the same.

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
