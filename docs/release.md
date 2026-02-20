# Release notes for demucs-mlx

## v1.0.0

MLX-native inference release. No PyTorch required at runtime.

- Pure MLX inference pipeline (Conv1d, Conv2d, Transformer, STFT/iSTFT)
- Custom fused Metal kernels: GroupNorm+GELU, GroupNorm+GLU, GLU, OLA, complex-to-interleaved
- Metal-free fallbacks for non-Apple platforms (Linux)
- Audio I/O via mlx-audio-io (replaces torchcodec/torchaudio/soundfile)
- STFT/iSTFT via mlx-spectro
- ~73x realtime on Apple Silicon — 2.6x faster than Demucs with PyTorch MPS
- Public Python API: `Separator`, `save_audio`, `list_models`
- CLI: `demucs-mlx` with threaded audio prefetch and parallel stem writing
- Native Conv1d resampling (replaces torchaudio resampling)
- Optional `[convert]` extra for one-time PyTorch weight conversion

### Differences from upstream Demucs

- Inference only — no training code
- Callbacks not yet supported in `Separator`
- Automatic resampling via mlx-audio-io when input sample rate differs from model (44100 Hz)
- PyTorch only needed for initial model weight conversion (`pip install 'demucs-mlx[convert]'`)
