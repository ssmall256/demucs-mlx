"""Compare previous and phased decoders on a stereo PCM16 WAV.

Run through metalq; use --input to point at a 44.1 kHz stereo WAV.
"""

import argparse
import wave

import numpy as np

from demucs_mlx.api import Separator
from demucs_mlx.mlx_layers import ConvTranspose1dNCL, ConvTranspose2dNCHW


def previous_1d(self, x):
    return self.conv(x.transpose(0, 2, 1)).transpose(0, 2, 1)


def previous_2d(self, x):
    return self.conv(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()
with wave.open(args.input, "rb") as input_wav:
    if (
        input_wav.getframerate() != 44_100
        or input_wav.getnchannels() != 2
        or input_wav.getsampwidth() != 2
    ):
        raise ValueError("Input must be 44.1 kHz stereo PCM16")
    frames = input_wav.getnframes()
    pcm = np.frombuffer(input_wav.readframes(frames), dtype="<i2")
audio = pcm.reshape(frames, 2).T.astype(np.float32) / 32768

current_1d = ConvTranspose1dNCL.__call__
current_2d = ConvTranspose2dNCHW.__call__
separator = Separator(seed=481)
try:
    ConvTranspose1dNCL.__call__ = previous_1d
    ConvTranspose2dNCHW.__call__ = previous_2d
    _, reference = separator.separate_tensor(audio)
    ConvTranspose1dNCL.__call__ = current_1d
    ConvTranspose2dNCHW.__call__ = current_2d
    _, phased = separator.separate_tensor(audio)
finally:
    ConvTranspose1dNCL.__call__ = current_1d
    ConvTranspose2dNCHW.__call__ = current_2d

print("## PCM16 WAV decoder parity")
print(f"**Input:** {frames} frames, seed=481")
print("| Stem | SNR | Peak error |")
print("|---|---:|---:|")
for stem, source in reference.items():
    want = source.astype(np.float64)
    error = want - phased[stem].astype(np.float64)
    snr = 10 * np.log10(np.sum(want * want) / max(np.sum(error * error), 1e-30))
    peak = np.max(np.abs(error))
    print(f"| {stem} | {snr:.2f} dB | {peak:.6g} |")
