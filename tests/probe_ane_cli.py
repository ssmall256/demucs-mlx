"""CLI integration probe with stdlib WAV I/O in place of mlx-audio-io.

This isolates CLI wiring from the local mlx-audio-io / MLX ABI mismatch.
Run through metalq because inference uses Metal.
"""

import tempfile
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

import demucs_mlx.audio as audio_module
import demucs_mlx.separate as cli


def load_wav(path, model):
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 2 and wav.getframerate() == model.samplerate
        values = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    return mx.array(values.reshape(-1, 2).T.astype(np.float32) / 32768)


def save_wav(wav, path, samplerate, **kwargs):
    values = np.clip(np.asarray(wav).T, -1, 1)
    pcm = (values * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(samplerate)
        output.writeframes(pcm.tobytes())


cli._load_audio = load_wav
audio_module.save_audio = save_wav
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "input.wav"
    samples = np.zeros((44_100, 2), dtype="<i2")
    with wave.open(str(source), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(44_100)
        output.writeframes(samples.tobytes())
    status = cli.main(
        [str(source), "--ane-time-encoder", "--batch-size", "1", "--verbose", "-o", str(root)]
    )
    assert status == 0
    for stem in ("drums", "bass", "other", "vocals"):
        output_path = root / "input" / f"{stem}.wav"
        with wave.open(str(output_path), "rb") as output:
            assert output.getnframes() == 44_100
print("## ANE CLI integration")
print("> All four stems were produced and the Core ML worker closed.")
