# Neural Engine waveform prototype

## Result

The opt-in `--ane-time-encoder` path sends the first `htdemucs` waveform convolution to the Apple Neural Engine while MLX evaluates the independent spectral encoder on the GPU. The remaining waveform layers stay on MLX. On the tested M4 Max, this gives essentially the same end-to-end throughput as the GPU default; it is a working, measured prototype.

The converter uses the official checkpoint through `get_restricted_demucs_model`, global waveform normalization, and an exact tiling of the first stride-4 convolution. The 343,980-sample input becomes 21 overlapping 16,384-sample convolutions in one fixed-shape Core ML model. The overlap preserves every output sample, including boundaries. The compiled model and manifest live under `~/.cache/demucs-mlx/ane/`; the manifest checks the validated MLX safetensors SHA-256 before inference. Conversion and inference require macOS. Only the default `htdemucs`, 7.8-second split segments, and batch sizes 1 or 2 are accepted.

```bash
uv sync --frozen --extra ane --extra ane-convert
metalq submit -w --no-env-sync -n demucs-ane-conv-convert -- python -m demucs_mlx.ane convert
demucs-mlx --ane-time-encoder song.wav
```

The project requires `mlx-audio-io` 1.3.20 or newer, and the lockfile selects 1.3.20. Its documented `tool.uv.extra-build-dependencies` setting in `pyproject.toml` builds the native extension against the locked MLX runtime, so file-based CLI use no longer needs a local sibling checkout.

The runtime needs only the `ane` extra after conversion. The equivalent API is `Separator(ane_time_encoder=True)`; use it as a context manager or call `close()` to stop its worker. `--verbose` prints execution, wait, and transfer time. Ordinary inference remains on MLX.

## Why the partition changed

The original plan requested normalization and all four waveform encoder stages in one full-length Core ML model. That model converted, but the `CPU_AND_NE` compute plan chose CPU for all **274 operations**. Replacing its first convolution with 21 exact tiles still led Core ML to choose CPU for all **317 operations** of the combined graph. The compiled diagnostic models remain in the same cache directory with the `htdemucs_time_encoder_b2` and `htdemucs_time_encoder_b2_tiled` names. These are anticipated placement results on this host; Core ML does not explain the scheduler's decision.

A shorter 16,384-sample, four-stage model placed all **265 operations** on the ANE. But running independent chunks changes the time-spanning GroupNorm statistics within the Demucs `DConv` blocks. Even with a 4,096-sample halo, the fourth-stage output was only **23.32 dB SNR** against the full-length MLX encoder. The exact convolution partition avoids that change: global normalization stays global, the convolution is mathematically equivalent, and all later time-coupled layers use the full signal on MLX.

The isolated convolution's compute plan reports **42 ANE operations** and **10 CPU operations**, with 45.3% of its estimated cost assigned to ANE. That percentage describes the isolated Core ML model, not the entire separation pipeline. The final compiled model is `htdemucs_time_conv_b2.mlmodelc`. Batch-one tails duplicate their input into batch two and discard the extra output.

The [john-rocky conversion](https://github.com/john-rocky/CoreML-Models/blob/5fa36bf4a82ccf710e3f47cfb073f3af0d568de2/conversion_scripts/convert_htdemucs.py#L27) exports the full batch-one model through ONNX in FP32; it does not report ANE placement. The [dexxdean conversion notes](https://huggingface.co/dexxdean/htdemucs-coreml/blob/main/CONVERSION_NOTES.md) document a full Core ML model and warn that ANE routing can produce incorrect output, recommending CPU/GPU for that model. These references prompted the additional placement and fidelity experiments above.

## Fidelity

The ANE convolution measured **68.69 dB SNR** and **0.00296 peak error** against the MLX convolution on deterministic random stereo input. A complete single-segment comparison measured 57.10–79.75 dB per-stem SNR, with peak errors at most 0.000126. The worker's batch-one padding produced identical values to the first item in a batch-two prediction, and `close()` stopped the worker thread.

The 30- and 60-second benchmark below uses default inference settings (one shift, 25% overlap, split mode, batch two) and the same fixed random seed for both paths. Complete stem comparisons gave:

| Input | Drums | Bass | Other | Vocals | Largest peak error |
|---|---:|---:|---:|---:|---:|
| 30 s | 63.70 dB | 71.19 dB | 72.30 dB | 61.79 dB | 0.0000619 |
| 60 s | 60.37 dB | 68.34 dB | 73.24 dB | 59.41 dB | 0.0000687 |

## Throughput

All GPU and ANE runs were submitted with `metalq submit -w` in job `mq-0cf032`. The sequence alternated GPU/ANE twice for each duration. Times include `Separator.separate_tensor` and materializing every stem, but exclude model loading and audio file I/O.

| Input | Pair | GPU | ANE | GPU / ANE | ANE execution | ANE wait | Transfer |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30 s | 1 | 0.696 s | 0.638 s | 1.09× | 0.050 s | 0.000 s | 0.004 s |
| 30 s | 2 | 0.644 s | 0.640 s | 1.01× | 0.039 s | 0.000 s | 0.003 s |
| 60 s | 1 | 1.177 s | 1.175 s | 1.00× | 0.060 s | 0.000 s | 0.006 s |
| 60 s | 2 | 1.189 s | 1.166 s | 1.02× | 0.060 s | 0.000 s | 0.006 s |

The first 30-second GPU run was slower than its repeat. Subsequent pairs differed by at most 0.023 s. The isolated offload is small relative to the rest of the pipeline; it shows no sustained throughput gain on this host.

## Reproduce the probes

```bash
metalq submit -w --no-env-sync -n demucs-ane-batch-placement -- python tests/probe_ane_placement.py
metalq submit -w --no-env-sync -n demucs-ane-chunk-placement -- python tests/probe_ane_chunk.py
metalq submit -w --no-env-sync -n demucs-ane-chunk-fidelity -- python tests/probe_ane_chunk_fidelity.py
metalq submit -w --no-env-sync -n demucs-ane-norm-conv-placement -- python tests/probe_ane_tiled_conv.py
metalq submit -w --no-env-sync -n demucs-ane-conv-parity -- python tests/probe_ane_parity.py
metalq submit -w --no-env-sync -n demucs-ane-stem-parity -- python tests/probe_ane_stems.py
metalq submit -w --no-env-sync -n demucs-ane-30-60-benchmark -- python tests/bench_ane_waveform.py
metalq submit -w --no-env-sync -n demucs-ane-cli-integration -- python tests/probe_ane_cli.py
```

Measurements used an Apple M4 Max with macOS 27, MLX 0.32.3, Core ML Tools 9.0, and PyTorch 2.7.1. The 16,384-sample isolated convolution was placed on ANE only at batch two; the batch-one probe chose CPU, which is why the runtime pads final batches.

## Real-audio CLI validation

The earlier published 1.3.19 source built with nanobind 2.15.0 against MLX 0.32.3 and failed to return MLX arrays from `load()`. [MLX v0.32.3 pins nanobind 3.0.1](https://github.com/ml-explore/mlx/blob/v0.32.3/CMakeLists.txt). Published [`mlx-audio-io` 1.3.20](https://pypi.org/project/mlx-audio-io/1.3.20/) documents the matching build dependency configuration now present in `pyproject.toml`. A forced reinstall from the PyPI source distribution with `uv sync --extra dev --extra ane --extra ane-convert --reinstall-package mlx-audio-io` replaced the local checkout installation. The installed package has no `direct_url.json`; its native build metadata reports `mlx-audio-io` 1.3.20, MLX 0.32.3, and nanobind 3.0.1.

With the published build, `metalq` job `mq-46f543` decoded a 1-second stereo WAV to an MLX array with shape `(44100, 2)`. Job `mq-9fa364` ran the ANE CLI on that WAV and wrote four 44,100-frame stems. Job `mq-3fe336` ran the ANE CLI on a 30-second stereo WAV with default inference settings and batch two, wrote four 1,323,000-frame stems, and reported four ANE predictions, 0.057 s execution, 0.000 s wait, and 0.014 s transfer. The default GPU CLI passed in job `mq-83f449`, writing four 44,100-frame stems. All Metal-bound runs used `metalq submit -w`.

The 1-second real-WAV CLI runs used the same seed. Comparing their written PCM16 stems with the GPU output as reference gave:

| Stem | SNR | Peak error |
|---|---:|---:|
| Drums | 54.70 dB | 0.0001526 |
| Bass | 58.52 dB | 0.0000305 |
| Other | 60.56 dB | 0.0001221 |
| Vocals | 46.70 dB | 0.0000305 |

The CLI and `Separator.separate_audio_file()` retain their actionable error if a mismatched native audio extension is installed outside this locked environment.
