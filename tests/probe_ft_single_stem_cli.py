"""Exercise the real CLI audio path for one fine-tuned stem via MetalQ."""

import tempfile
import wave
from pathlib import Path

import numpy as np

from demucs_mlx.separate import main

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "input.wav"
    samples = np.zeros((44_100, 2), dtype="<i2")
    with wave.open(str(source), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(44_100)
        output.writeframes(samples.tobytes())

    assert main([str(source), "-n", "htdemucs_ft", "--stem", "vocals", "-o", str(root)]) == 0
    outputs = list((root / "input").glob("*.wav"))
    assert [path.name for path in outputs] == ["vocals.wav"], outputs
    with wave.open(str(outputs[0]), "rb") as result:
        assert result.getnframes() == 44_100

print("## Fine-tuned single-stem CLI")
print("> ✅ Wrote only `vocals.wav`, with the expected 44,100 frames.")
