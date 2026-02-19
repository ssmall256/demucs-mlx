# Release notes for demucs-mlx

## v1.0.0

MLX-native inference release. No PyTorch required at runtime.

- Pure MLX inference pipeline (Conv1d, Conv2d, Transformer, STFT/iSTFT)
- Custom fused Metal kernels: GroupNorm+GELU, GroupNorm+GLU, GLU, OLA, complex-to-interleaved
- Metal-free fallbacks for non-Apple platforms (Linux)
- Audio I/O via mlx-audio-io (replaces torchcodec/torchaudio/soundfile)
- STFT/iSTFT via mlx-spectro
- ~67x realtime on Apple Silicon — 2.4x faster than Demucs with PyTorch MPS
- Public Python API: `Separator`, `save_audio`, `list_models`
- CLI: `demucs-mlx` with threaded audio prefetch and parallel stem writing
- Native Conv1d resampling (replaces torchaudio resampling)
- Optional `[convert]` extra for one-time PyTorch weight conversion

### Differences from upstream Demucs

- Inference only — no training code
- Callbacks not yet supported in `Separator`
- Input audio must match model sample rate (44100 Hz) — automatic resampling not yet wired up
- PyTorch only needed for initial model weight conversion (`pip install 'demucs-mlx[convert]'`)
