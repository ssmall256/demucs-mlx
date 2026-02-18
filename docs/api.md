# Demucs-MLX API

## Quick start

```python
from demucs_mlx import Separator, save_audio, list_models

# List available models
print(list_models())

# Separate an audio file
separator = Separator(model="htdemucs")
origin, stems = separator.separate_audio_file("song.wav")

# Save stems
for name, audio in stems.items():
    save_audio(audio, f"{name}.wav", samplerate=separator.samplerate)
```

## `class Separator`

### Parameters

- **model** (`str`): Model name. Default `"htdemucs"`. See `list_models()` for options.
- **shifts** (`int`): Random time shifts for equivariance averaging. Default `1`. Set to `0` to disable.
- **overlap** (`float`): Overlap ratio between segments. Default `0.25`. Must be in `[0, 1)`.
- **split** (`bool`): Whether to split audio into segments. Default `True`.
- **segment** (`float | None`): Segment length in seconds. Default `None` (uses model default).
- **progress** (`bool`): Show progress bar. Default `False`.

### Properties

- **`samplerate`** (`int`): Sample rate the model expects.
- **`audio_channels`** (`int`): Number of audio channels the model expects.
- **`model`**: The underlying MLX model instance.

### `update_parameter(**kwargs)`

Update separation parameters after initialization. Accepts the same keyword arguments as the constructor (except `model`).

### `separate_tensor(wav, *, return_mx=False)`

Separate a waveform tensor.

**Parameters:**
- **wav**: Waveform with shape `(channels, time)`. Accepts `numpy.ndarray` or `mlx.core.array`.
- **return_mx** (`bool`): If `True`, return MLX arrays instead of numpy. Default `False`.

**Returns:** `(origin, stems)` where `origin` is the input waveform and `stems` is a `dict[str, array]` mapping source names to separated waveforms.

### `separate_audio_file(path, *, return_mx=False)`

Separate an audio file. Reads the file using `mlx-audio-io`.

**Parameters:**
- **path** (`str | Path`): Path to the audio file.
- **return_mx** (`bool`): If `True`, return MLX arrays instead of numpy. Default `False`.

**Returns:** Same as `separate_tensor`.

**Note:** The audio file's sample rate must match the model's sample rate (44100 Hz for all current models). Resampling is not yet supported.

## `save_audio(wav, path, samplerate, ...)`

Save audio to a file.

**Parameters:**
- **wav**: Audio data. Accepts `numpy.ndarray`, `mlx.core.array`, or `torch.Tensor`.
- **path** (`str | Path`): Output file path (`.wav` supported).
- **samplerate** (`int`): Sample rate.
- **clip** (`str`): Clipping prevention mode: `"rescale"` (default), `"clamp"`, `"tanh"`, or `"none"`.
- **bits_per_sample** (`int`): Bit depth: `16` (default), `24`, or `32`.
- **as_float** (`bool`): Save as 32-bit float. Default `False`.

## `list_models()`

Returns a dict with keys `"single"` and `"bag"`, each containing a list of model name strings.
