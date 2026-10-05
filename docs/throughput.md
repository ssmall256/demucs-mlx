# Throughput

RTFx is audio seconds divided by wall seconds; larger is faster. All figures are
for `htdemucs` with default settings on an Apple M4 Max (40-core GPU, 128 GB)
with MLX 0.32.3, measured in October 2026. Other machines will differ.

## What the numbers mean

| Measurement | Definition | Result |
|---|---|---:|
| Warm call | A separation after the first one in a process, tensor in to stems out | about 114x |
| First call | The first separation in a new process; includes graph compilation, Metal pipeline setup and buffer allocation. Model loading is excluded | about 99-107x |
| PyTorch MPS | `demucs` 4.1.0 on PyTorch 2.14.1, same input and boundary, warmed | 42x |
| PyTorch MPS, STFT on GPU | Same, with Demucs patched to keep STFT/iSTFT on MPS | 50x |
| PyTorch CPU | Same, on CPU | 6x |

The PyTorch comparison alternated implementations on the same 120-second stereo
input and took the median of four warmed calls each. Upstream Demucs runs its
STFT and iSTFT on the CPU when the model is on MPS, which costs it about 18% of
each call. PyTorch has been able to run them on MPS since 2.3.0; with Demucs
patched to do so (PyTorch's own ops or `mps-spectro`) it reaches 50x with
output unchanged to within -120 dB.

## What determines it

- **Compilation.** The GPU forward is compiled per chunk shape on first use
  (`DEMUCS_MLX_COMPILE_FORWARD`, on by default). The first call pays for it;
  later calls with the same shape do not.
- **Batch size.** `auto` picks a batch per machine and spreads a track's chunks
  evenly across batches, so a track needs fewer distinct shapes.
- **Buffer cache.** MLX reuses GPU buffers between calls. Clearing the cache
  between tracks costs 75-135 ms per call.
- **Sustained load.** Continuous back-to-back separation heats the machine, and
  the system eventually reduces throughput. `bench_rtfx.py` reports the last-five
  median so you can see it.
- **Other GPU work.** Anything else using the GPU shares it.

## Reproduce

```bash
python tests/bench_rtfx.py --audio song.m4a --processes 3 --calls 10
python tests/bench_rtfx.py --seconds 120 --processes 1 --calls 30   # sustained
```

The script's docstring defines each measurement. Run GPU measurements one at a
time and close GPU-heavy applications first.

## Things that were tried and not adopted

- Treating frequency 2D convolutions as 1D: transposed convolutions became
  3-12x slower.
- Approximate GELU: output agreement fell to about 41 dB with no consistent
  speedup.
- Half precision for the whole model: slower end to end, 38-42 dB agreement.
  Half precision is used only inside the attention kernel, where it is within
  0.5 dB of float32 and about 3% faster.
- A Neural Engine offload of the first waveform convolution: same end-to-end
  throughput as the GPU path, so it was removed.
