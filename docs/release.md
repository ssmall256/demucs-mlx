# Release notes for Demucs-MLX

## V4.1.0a3

MLX-native inference release. No PyTorch required at runtime.

- Pure MLX inference pipeline (Conv1d, Conv2d, Transformer, STFT/iSTFT)
- Custom fused Metal kernels: GroupNorm+GELU, GroupNorm+GLU, GLU, OLA, complex-to-interleaved
- Metal-free fallbacks for non-Apple platforms (Linux)
- Audio I/O via mlx-audio-io (replaces torchcodec/torchaudio/soundfile)
- ~67x realtime on Apple Silicon
- Public Python API: `Separator`, `save_audio`, `list_models`
- CLI: `demucs-mlx` with threaded audio prefetch and parallel stem writing
- Reduced redundant `mx.eval()` synchronization points
- Native Conv1d resampling (replaces torchaudio resampling)

### Breaking changes from upstream Demucs

- `device` parameter only accepts `"mlx"` (no CUDA/MPS/CPU)
- Callbacks not yet supported in `Separator`
- Audio resampling not yet supported (input must match model sample rate)
- PyTorch only needed for initial model weight conversion (lazy import)

## V4.0.1, 8th of September 2023

**From this version, Python 3.7 is no longer supported.**

Various improvements by @CarlGao4. Support for `segment` param inside of HTDemucs model.

## V4.0.0, 7th of December 2022

Adding hybrid transformer Demucs model.

Added experimental 6 sources model `htdemucs_6s` (`drums`, `bass`, `other`, `vocals`, `piano`, `guitar`).
