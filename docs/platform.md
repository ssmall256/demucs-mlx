# Platform notes

This project is uv-first. All instructions below use uv and the `uv.lock` workflow.

## macOS

- Apple Silicon is supported via MLX.
- Install FFmpeg (needed for mp3 decoding).

Typical flow:

```bash
uv lock
uv sync
uv run demucs-mlx /path/to/audio.wav
```

## Linux

- Python >= 3.10 required.
- Install FFmpeg via your distribution package manager.

```bash
uv lock
uv sync
uv run demucs-mlx /path/to/audio.wav
```

## Windows

- Python >= 3.10 required.
- Install FFmpeg (use a trusted Windows package manager or official builds).

```powershell
uv lock
uv sync
uv run demucs-mlx "C:\\path\\to\\audio.wav"
```
