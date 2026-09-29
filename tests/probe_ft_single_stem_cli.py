"""Check real fine-tuned CLI single-stem outputs against the full bag."""

import tempfile
import wave
from pathlib import Path

import numpy as np

from demucs_mlx.separate import main

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "input.wav"
    times = np.arange(44_100) / 44_100
    mono = np.round(12000 * np.sin(2 * np.pi * 440 * times)).astype("<i2")
    samples = np.stack((mono, mono // 2), axis=1)
    with wave.open(str(source), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(44_100)
        output.writeframes(samples.tobytes())

    common = [str(source), "-n", "htdemucs_ft", "--seed", "481"]
    assert main([*common, "-o", str(root / "full")]) == 0
    reference_dir = root / "full" / "input"

    for stem in ("drums", "bass", "vocals"):
        assert main([*common, "--stem", stem, "-o", str(root / stem)]) == 0
        outputs = list((root / stem / "input").glob("*.wav"))
        assert [path.name for path in outputs] == [f"{stem}.wav"], outputs
        with wave.open(str(outputs[0]), "rb") as selected_file:
            assert selected_file.getnframes() == 44_100
            selected = selected_file.readframes(44_100)
        with wave.open(str(reference_dir / f"{stem}.wav"), "rb") as full_file:
            assert selected == full_file.readframes(44_100)

print("## Fine-tuned single-stem CLI")
print("> ✅ `drums`, `bass`, and `vocals` each wrote one byte-identical 44,100-frame stem.")
